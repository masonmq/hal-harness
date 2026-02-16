#!/bin/bash
set -euo pipefail

echo "[replicatorbench] setup.sh starting..."

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_JSON="${BENCH_DIR}/tasks.json"

if [[ ! -f "${TASKS_JSON}" ]]; then
  echo "[replicatorbench] ERROR: tasks.json not found at: ${TASKS_JSON}"
  exit 1
fi

# --- workspace inside HAL environment ---
ROOT="/root/environment/workspace"
CAPS_DIR="${ROOT}/capsules"
mkdir -p "${CAPS_DIR}"

KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  python3 \
  python3-pip \
  unzip \
  tar

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade gdown

download_gdrive_folder() {
  local url="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  gdown --no-cookies --folder "${url}" -O "${out_dir}"
}

download_gdrive_file() {
  local url_or_id="$1"
  local out_path="$2"

  local id=""
  if [[ "${url_or_id}" =~ ^[0-9A-Za-z_-]{20,}$ ]]; then
    id="${url_or_id}"
  else
    id="$(python3 - <<'PY' "${url_or_id}"
import re,sys
u=sys.argv[1]
m = re.search(r"/file/d/([^/]+)", u) or re.search(r"[?&]id=([^&]+)", u)
print(m.group(1) if m else "")
PY
)"
  fi

  if [[ -z "${id}" ]]; then
    echo "[replicatorbench] ERROR: cannot parse Google Drive file id from: ${url_or_id}"
    return 1
  fi

  local base="https://drive.google.com/uc?export=download&id=${id}"
  local cookie="/tmp/gdrive_cookie_$$.txt"
  local tmp="${out_path}.tmp"

  rm -f "${cookie}" "${tmp}"

  curl -L -sS -c "${cookie}" -o "${tmp}" "${base}"

  if head -c 2 "${tmp}" | grep -q "PK"; then
    mv -f "${tmp}" "${out_path}"
    rm -f "${cookie}"
    return 0
  fi

  if grep -q "Google Drive can't scan this file for viruses" "${tmp}"; then
    local confirm uuid
    confirm="$(sed -n 's/.*name="confirm" value="\([^"]*\)".*/\1/p' "${tmp}" | head -n 1)"
    uuid="$(sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' "${tmp}" | head -n 1)"

    if [[ -z "${confirm}" ]]; then
      echo "[replicatorbench] ERROR: could not parse confirm token from virus-scan warning page."
      rm -f "${tmp}" "${cookie}"
      return 1
    fi

    echo "[replicatorbench]   virus-scan warning detected; downloading from drive.usercontent.google.com..."
    local dl="https://drive.usercontent.google.com/download?export=download&id=${id}&confirm=${confirm}"
    if [[ -n "${uuid}" ]]; then
      dl="${dl}&uuid=${uuid}"
    fi

    curl -L --fail -b "${cookie}" --progress-bar -o "${out_path}" "${dl}"
    rm -f "${tmp}" "${cookie}"
    return 0
  fi

  echo "[replicatorbench] ERROR: Google Drive returned HTML instead of a file, and it was not the virus-scan warning page."
  echo "[replicatorbench] First lines:"
  head -n 20 "${tmp}" | sed 's/^/[replicatorbench]   /'
  rm -f "${tmp}" "${cookie}"
  return 1
}

extract_zip_capsule() {
  local zip_path="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local tmp="/tmp/capsule_unzip_$$"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  unzip -q "${zip_path}" -d "${tmp}"

  local top_count
  top_count="$(find "${tmp}" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"

  if [[ "${top_count}" == "1" ]] && [[ -d "$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)" ]]; then
    local topdir
    topdir="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    shopt -s dotglob
    cp -R "${topdir}"/* "${out_dir}"/
    shopt -u dotglob
  else
    shopt -s dotglob
    cp -R "${tmp}"/* "${out_dir}"/
    shopt -u dotglob
  fi

  rm -rf "${tmp}"

  if compgen -G "${out_dir}/*.pdf.pdf" > /dev/null; then
    for f in "${out_dir}"/*.pdf.pdf; do
      mv -f "${f}" "${f%.pdf}"
    done
  fi
}

download_tarball_capsule() {
  local url="$1"
  local sha="$2"
  local out_dir="$3"
  local tgz_path="$4"

  echo "[replicatorbench] downloading tarball: ${url}"
  curl -L --fail -o "${tgz_path}" "${url}"

  if [[ -n "${sha}" && "${sha}" != "null" ]]; then
    echo "[replicatorbench] verifying sha256..."
    echo "${sha}  ${tgz_path}" | sha256sum -c -
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  tar -xzf "${tgz_path}" -C "${out_dir}" --strip-components=1
}

# the directory where agents will write all stage outputs
prepare_work_dir() {
  local task_id="$1"
  local work_dir="${ROOT}/${task_id}"

  rm -rf "${work_dir}"
  mkdir -p "${work_dir}"

  echo "/root/environment/workspace/capsules/${task_id}" > "${work_dir}/CAPSULE_PATH.txt"
}

make_capsule_readonly() {
  local capsule_dir="$1"
  chmod -R a-w "${capsule_dir}" || true
}

# Parse tasks.json and download capsules
while IFS=$'\t' read -r task_id capsule_type capsule_url capsule_sha256; do
  if [[ -z "${task_id}" ]]; then
    continue
  fi
  if [[ -z "${capsule_url}" ]]; then
    echo "[replicatorbench] ${task_id}: no capsule_url (skipping download)"
    continue
  fi

  out_dir="${CAPS_DIR}/${task_id}"

  if [[ "${capsule_type}" == "gdrive_zip" ]]; then
    echo "[replicatorbench] ${task_id}: downloading ZIP capsule..."
    zip_path="${CAPS_DIR}/${task_id}.zip"
    download_gdrive_file "${capsule_url}" "${zip_path}"
    extract_zip_capsule "${zip_path}" "${out_dir}"
    if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
      rm -f "${zip_path}"
    fi

  elif [[ "${capsule_type}" == "gdrive_folder" ]]; then
    echo "[replicatorbench] ${task_id}: downloading folder capsule..."
    download_gdrive_folder "${capsule_url}" "${out_dir}"

  else
    echo "[replicatorbench] ${task_id}: downloading tarball capsule..."
    tgz_path="${CAPS_DIR}/${task_id}.tar.gz"
    download_tarball_capsule "${capsule_url}" "${capsule_sha256}" "${out_dir}" "${tgz_path}"
    if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
      rm -f "${tgz_path}"
    fi
  fi

  # create the working directory
  prepare_work_dir "${task_id}"

  make_capsule_readonly "${out_dir}"

  echo "[replicatorbench] ${task_id}: capsule ready at ${out_dir}"
  echo "[replicatorbench] ${task_id}: work dir ready at ${ROOT}/${task_id}"
done < <(python3 - "${TASKS_JSON}" <<'PY'
import json, sys
tasks_path = sys.argv[1]
with open(tasks_path, "r") as f:
    obj = json.load(f)
tasks = obj["tasks"] if isinstance(obj, dict) and "tasks" in obj else obj
if not isinstance(tasks, list):
    raise SystemExit("tasks.json must be a dict with key 'tasks' holding a list.")
for t in tasks:
    task_id = str(t.get("task_id", "")).strip()
    ctype = (t.get("capsule_type") or "").strip()
    url = (t.get("capsule_url") or "").strip()
    sha = t.get("capsule_sha256", None)
    sha_s = "" if sha is None else str(sha).strip()
    print(f"{task_id}\t{ctype}\t{url}\t{sha_s}")
PY
)

echo "[replicatorbench] setup.sh complete."
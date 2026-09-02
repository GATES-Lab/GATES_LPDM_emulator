#!/bin/bash
# Automates transferring a met zarr archive from Jasmin to Isambard-AI:
#   1. ssh to Jasmin and tar the zarr directory (slow: ~15-20 min)
#   2. rsync the tar from Jasmin to Isambard
#   3. extract the tar into a zarr on Isambard
#   4. verify size/file-count match between Jasmin and Isambard
#   5. (optional) delete the tar(s), and optionally the source zarr on Jasmin
#
# Usage:
#   ./scripts/transfer_met_data.sh REGION YEAR [--skip-tar] [--skip-rsync] [--skip-extract] \
#                                      [--cleanup] [--cleanup-jasmin-zarr] [-y]
#
# Example:
#   ./scripts/transfer_met_data.sh CHINA 2015
#
# NOTE: The extraction step should be run on a compute/interactive node, not the
# login node. If not already inside an allocation (no $SLURM_JOB_ID), this script
# will warn you and ask for confirmation before extracting.
#
# Configuration (override via environment variables):
#   JASMIN_USER            (default: jeff)
#   JASMIN_HOST             (default: hpxfer3.jasmin.ac.uk)
#   JASMIN_ZARR_BASE        (default: /gws/ssde/j25b/acrg/elenafi/satellite_met_zarr)
#   JASMIN_TAR_BASE         (default: /gws/ssde/j25b/acrg/jeff/satellite_met/zarrs)
#   ISAMBARD_MET_BASE       (default: /projects/b5bn/data/met_archive_zarr)

set -euo pipefail

# --- Args --------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 REGION YEAR [--skip-tar] [--skip-rsync] [--skip-extract] [--cleanup] [--cleanup-jasmin-zarr] [-y]" >&2
  exit 1
fi

REGION="$1"
YEAR="$2"
shift 2

SKIP_TAR=0
SKIP_RSYNC=0
SKIP_EXTRACT=0
CLEANUP=0
CLEANUP_JASMIN_ZARR=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tar) SKIP_TAR=1 ;;
    --skip-rsync) SKIP_RSYNC=1 ;;
    --skip-extract) SKIP_EXTRACT=1 ;;
    --cleanup) CLEANUP=1 ;;
    --cleanup-jasmin-zarr) CLEANUP_JASMIN_ZARR=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# --- Config --------------------------------------------------------------
JASMIN_USER="${JASMIN_USER:-jeff}"
JASMIN_HOST="${JASMIN_HOST:-hpxfer3.jasmin.ac.uk}"
JASMIN_ZARR_BASE="${JASMIN_ZARR_BASE:-/gws/ssde/j25b/acrg/elenafi/satellite_met_zarr}"
JASMIN_TAR_BASE="${JASMIN_TAR_BASE:-/gws/ssde/j25b/acrg/jeff/satellite_met/zarrs}"
ISAMBARD_MET_BASE="${ISAMBARD_MET_BASE:-/projects/b5bn/data/met_archive_zarr}"

ZARR_NAME="${REGION}_Met_${YEAR}.zarr"
TAR_NAME="${REGION}_Met_${YEAR}.tar"

JASMIN_ZARR_DIR="${JASMIN_ZARR_BASE}/${REGION}"
JASMIN_TAR_PATH="${JASMIN_TAR_BASE}/${TAR_NAME}"
ISAMBARD_DEST_DIR="${ISAMBARD_MET_BASE}/${REGION}"
ISAMBARD_TAR_PATH="${ISAMBARD_DEST_DIR}/${TAR_NAME}"
ISAMBARD_ZARR_PATH="${ISAMBARD_DEST_DIR}/${ZARR_NAME}"

JASMIN_SSH="${JASMIN_USER}@${JASMIN_HOST}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

confirm() {
  local prompt="$1"
  if [[ "${ASSUME_YES}" -eq 1 ]]; then
    return 0
  fi
  read -r -p "${prompt} [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]]
}

# --- 1. Create tar on Jasmin ---------------------------------------------
if [[ "${SKIP_TAR}" -eq 1 ]]; then
  log "Skipping tar creation on Jasmin (--skip-tar)."
else
  log "Checking for existing tar on Jasmin: ${JASMIN_TAR_PATH}"
  if ssh "${JASMIN_SSH}" "test -f '${JASMIN_TAR_PATH}'"; then
    log "Tar already exists on Jasmin, skipping creation. Remove it manually to force a rebuild."
  else
    log "Creating tar on Jasmin (this can take 15-20 minutes)..."
    ssh "${JASMIN_SSH}" \
      "mkdir -p '${JASMIN_TAR_BASE}' && tar -cf '${JASMIN_TAR_PATH}' -C '${JASMIN_ZARR_DIR}' '${ZARR_NAME}'"
    log "Tar created on Jasmin: ${JASMIN_TAR_PATH}"
  fi
fi

# --- 2. rsync tar from Jasmin to Isambard --------------------------------
if [[ "${SKIP_RSYNC}" -eq 1 ]]; then
  log "Skipping rsync (--skip-rsync)."
else
  mkdir -p "${ISAMBARD_DEST_DIR}"
  log "Rsyncing tar from Jasmin to ${ISAMBARD_DEST_DIR}..."
  rsync -avP "${JASMIN_SSH}:${JASMIN_TAR_PATH}" "${ISAMBARD_DEST_DIR}/"
  log "Rsync complete: ${ISAMBARD_TAR_PATH}"
fi

# --- 3. Extract tar on Isambard -------------------------------------------
if [[ "${SKIP_EXTRACT}" -eq 1 ]]; then
  log "Skipping extraction (--skip-extract)."
else
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    log "WARNING: no SLURM_JOB_ID detected - you do not appear to be on a compute/interactive node."
    log "Recommended: srun --gpus=1 --reservation=interactive --pty bash -i"
    if ! confirm "Extract on this (possibly login) node anyway?"; then
      log "Aborting before extraction. Re-run with --skip-tar --skip-rsync from an interactive node to resume."
      exit 1
    fi
  fi
  log "Extracting ${ISAMBARD_TAR_PATH}..."
  tar -xf "${ISAMBARD_TAR_PATH}" -C "${ISAMBARD_DEST_DIR}" \
    --checkpoint=10000 --checkpoint-action=echo='Extracted %u files'
  log "Extraction complete: ${ISAMBARD_ZARR_PATH}"
fi

# --- 4. Verify size/file count match --------------------------------------
if [[ -d "${ISAMBARD_ZARR_PATH}" ]]; then
  log "Verifying zarr consistency between Jasmin and Isambard..."

  ISAMBARD_SIZE=$(du -sh "${ISAMBARD_ZARR_PATH}" | cut -f1)
  ISAMBARD_COUNT=$(find "${ISAMBARD_ZARR_PATH}" -type f | wc -l)

  read -r JASMIN_SIZE JASMIN_COUNT <<< "$(ssh "${JASMIN_SSH}" \
    "du -sh '${JASMIN_ZARR_DIR}/${ZARR_NAME}' | cut -f1; find '${JASMIN_ZARR_DIR}/${ZARR_NAME}' -type f | wc -l" \
    | paste -sd' ')"

  log "Jasmin:   size=${JASMIN_SIZE}, files=${JASMIN_COUNT}"
  log "Isambard: size=${ISAMBARD_SIZE}, files=${ISAMBARD_COUNT}"

  if [[ "${JASMIN_COUNT}" != "${ISAMBARD_COUNT}" ]]; then
    log "WARNING: file counts differ between Jasmin and Isambard!"
  else
    log "File counts match."
  fi
else
  log "Zarr not found at ${ISAMBARD_ZARR_PATH}, skipping verification."
fi

# --- 5. Cleanup -------------------------------------------------------------
if [[ "${CLEANUP}" -eq 1 ]]; then
  if [[ -f "${ISAMBARD_TAR_PATH}" ]] && confirm "Delete tar on Isambard (${ISAMBARD_TAR_PATH})?"; then
    rm -f "${ISAMBARD_TAR_PATH}"
    log "Deleted ${ISAMBARD_TAR_PATH}"
  fi

  if confirm "Delete tar on Jasmin (${JASMIN_TAR_PATH})?"; then
    ssh "${JASMIN_SSH}" "rm -f '${JASMIN_TAR_PATH}'"
    log "Deleted ${JASMIN_TAR_PATH} on Jasmin"
  fi

  if [[ "${CLEANUP_JASMIN_ZARR}" -eq 1 ]] && confirm "Delete source zarr on Jasmin (${JASMIN_ZARR_DIR}/${ZARR_NAME})? This is IRREVERSIBLE."; then
    ssh "${JASMIN_SSH}" "rm -rf '${JASMIN_ZARR_DIR}/${ZARR_NAME}'"
    log "Deleted ${JASMIN_ZARR_DIR}/${ZARR_NAME} on Jasmin"
  fi
else
  log "Cleanup skipped (pass --cleanup to remove tars, add --cleanup-jasmin-zarr to also offer deleting the Jasmin source zarr)."
fi

log "Done."

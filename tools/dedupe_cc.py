"""Find and remove/move duplicate cc_*.png images within each folder.

Two images count as the same when their files are byte-identical (SHA-256). Only images in the
same folder are compared. In each group of duplicates the lowest-numbered file is kept
(cc_1 before cc_2 before cc_10); the rest are the duplicates.

Dry run by default - it only prints what it would do. Pass --delete to remove the duplicates, or
--move to move them to <root>/dedupe_images/<folder>/<name>, where <folder> is the folder's path
relative to <root>. The batch container's folder scan ignores "dedupe_images" (like "ref"), and so
does this script.

    python dedupe_cc.py <root folder>            # show duplicates
    python dedupe_cc.py <root folder> --move     # move them into <root>/dedupe_images/<folder>/
    python dedupe_cc.py <root folder> --delete   # remove them
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict

CC_RE = re.compile(r"^cc_.*\.png$", re.IGNORECASE)
DEDUPE_DIR = "dedupe_images"


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def move_target(root, dirpath, name):
    """<root>/dedupe_images/<dirpath relative to root>/name, suffixed (_1, _2, ...) if already taken."""
    target_dir = os.path.normpath(os.path.join(root, DEDUPE_DIR, os.path.relpath(dirpath, root)))
    os.makedirs(target_dir, exist_ok=True)
    stem, ext = os.path.splitext(name)
    dst, n = os.path.join(target_dir, name), 1
    while os.path.exists(dst):
        dst = os.path.join(target_dir, f"{stem}_{n}{ext}")
        n += 1
    return dst


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate cc_*.png images within each folder.")
    parser.add_argument("root", help="folder to scan (its subfolders are scanned too)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--delete", action="store_true", help="delete the duplicates (default: dry run)")
    action.add_argument("--move", action="store_true", help=f"move the duplicates into <root>/{DEDUPE_DIR}/<folder>/")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a folder: {args.root}")

    scanned = removed = 0
    for dirpath, dirnames, filenames in os.walk(args.root):
        dirnames[:] = [d for d in dirnames if d.lower() != DEDUPE_DIR]  # don't re-scan moved duplicates
        images = sorted((f for f in filenames if CC_RE.match(f)), key=natural_key)
        if len(images) < 2:
            scanned += len(images)
            continue
        scanned += len(images)

        groups = defaultdict(list)
        for name in images:
            groups[file_hash(os.path.join(dirpath, name))].append(name)

        for names in groups.values():
            keep, dupes = names[0], names[1:]
            for name in dupes:
                rel = os.path.relpath(os.path.join(dirpath, name), args.root)
                src = os.path.join(dirpath, name)
                if args.delete:
                    print(f"delete  {rel}  (same as {keep})")
                    os.remove(src)
                elif args.move:
                    dst = move_target(args.root, dirpath, name)
                    print(f"move  {rel} -> {os.path.relpath(dst, args.root)}  (same as {keep})")
                    shutil.move(src, dst)
                else:
                    print(f"duplicate  {rel}  (same as {keep})")
                removed += 1

    verb = "Deleted" if args.delete else "Moved" if args.move else "Found"
    print(f"\nScanned {scanned} cc_*.png image(s). {verb} {removed} duplicate(s).")
    if removed and not (args.delete or args.move):
        print(f"Run again with --move (into <root>/{DEDUPE_DIR}/) or --delete to handle them.")


if __name__ == "__main__":
    main()

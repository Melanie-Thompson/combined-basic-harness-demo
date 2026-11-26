"""Package registry — stores and serves signed tool packages.

The registry holds tool definitions, source code, and signatures. When a
harness requests a package, the registry returns the full manifest (definition
+ source + signature) ready for installation.

Packages are loaded from the packages/ directory at startup. Each package
is a directory containing:
    definition.json  — the tool definition
    source.py        — the tool source code
    (signature is computed on demand from the publisher's key)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.signing import KeyStore, PackageSignature

log = logging.getLogger("registry")


@dataclass
class Package:
    name: str
    publisher: str
    definition: dict[str, Any]
    source_code: str
    signature: PackageSignature | None = None


class PackageRegistry:
    """Stores tool packages and serves them to agent harnesses."""

    def __init__(self, packages_dir: str | Path = "packages", key_store: KeyStore | None = None):
        self._dir = Path(packages_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._key_store = key_store or KeyStore()
        self._packages: dict[str, Package] = {}

    def load_all(self) -> int:
        """Load all packages from the packages directory. Returns count loaded."""
        count = 0
        for pkg_dir in sorted(self._dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            defn_path = pkg_dir / "definition.json"
            source_path = pkg_dir / "source.py"
            if not defn_path.exists() or not source_path.exists():
                log.warning("skipping %s: missing definition.json or source.py", pkg_dir.name)
                continue

            definition = json.loads(defn_path.read_text())
            source_code = source_path.read_text()
            name = definition.get("name", pkg_dir.name)

            meta_path = pkg_dir / "meta.json"
            publisher = ""
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                publisher = meta.get("publisher", "")

            sig = None
            if publisher:
                try:
                    sig = self._key_store.sign(publisher, definition, source_code)
                except FileNotFoundError:
                    log.warning("no signing key for publisher '%s', package '%s' will be unsigned",
                                publisher, name)

            self._packages[name] = Package(
                name=name,
                publisher=publisher,
                definition=definition,
                source_code=source_code,
                signature=sig,
            )
            log.info("loaded package '%s' (publisher: %s, signed: %s)",
                     name, publisher or "none", sig is not None)
            count += 1
        return count

    def get(self, name: str) -> Package | None:
        return self._packages.get(name)

    def list_packages(self) -> list[dict[str, Any]]:
        """List all packages (definition + publisher, no source code)."""
        return [
            {
                "name": pkg.name,
                "publisher": pkg.publisher,
                "definition": pkg.definition,
                "signed": pkg.signature is not None,
            }
            for pkg in self._packages.values()
        ]

    def get_manifest(self, name: str) -> dict[str, Any] | None:
        """Get the full installable manifest for a package."""
        pkg = self._packages.get(name)
        if pkg is None:
            return None
        manifest: dict[str, Any] = {
            "definition": pkg.definition,
            "source_code": pkg.source_code,
            "publisher": pkg.publisher,
        }
        if pkg.signature:
            manifest["signature"] = pkg.signature.to_dict()
        return manifest

    def add_package(
        self,
        definition: dict,
        source_code: str,
        publisher: str = "",
        persist: bool = True,
    ) -> Package:
        """Add a package to the registry (and optionally to disk)."""
        name = definition.get("name", "")
        if not name:
            raise ValueError("definition must have a name")

        sig = None
        if publisher:
            try:
                sig = self._key_store.sign(publisher, definition, source_code)
            except FileNotFoundError:
                log.warning("no signing key for '%s', package will be unsigned", publisher)

        pkg = Package(
            name=name,
            publisher=publisher,
            definition=definition,
            source_code=source_code,
            signature=sig,
        )
        self._packages[name] = pkg

        if persist:
            pkg_dir = self._dir / name
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (pkg_dir / "definition.json").write_text(
                json.dumps(definition, indent=2)
            )
            (pkg_dir / "source.py").write_text(source_code)
            if publisher:
                (pkg_dir / "meta.json").write_text(
                    json.dumps({"publisher": publisher}, indent=2)
                )

        log.info("added package '%s' (publisher: %s, signed: %s)",
                 name, publisher or "none", sig is not None)
        return pkg

    def remove_package(self, name: str) -> bool:
        if name not in self._packages:
            return False
        del self._packages[name]
        log.info("removed package '%s' from registry", name)
        return True

    @property
    def count(self) -> int:
        return len(self._packages)

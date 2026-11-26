"""Package signing — the publisher side.

Generates Ed25519 keypairs, signs tool packages (definition + source code),
and manages the publisher's key store.

The canonical payload matches the harness's verification:
SHA-256(canonical_definition_json) || SHA-256(source_code)
"""

import hashlib
import json
import logging
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
)

log = logging.getLogger("signing")


@dataclass
class PackageSignature:
    publisher: str
    public_key: str
    signature: str
    payload_hash: str

    def to_dict(self) -> dict:
        return {
            "publisher": self.publisher,
            "public_key": self.public_key,
            "signature": self.signature,
            "payload_hash": self.payload_hash,
        }


def generate_keypair() -> tuple[str, str]:
    """Generate an Ed25519 keypair. Returns (private_key_pem, public_key_pem)."""
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _canonical_definition(defn: dict) -> str:
    """Canonical JSON for the tool definition — matches the harness exactly."""
    canonical = {
        "name": defn.get("name", ""),
        "description": defn.get("description", ""),
        "inputSchema": defn.get("inputSchema", {}),
        "kind": defn.get("kind", "processor"),
        "placement": defn.get("placement", "local"),
        "capabilities": defn.get("capabilities", []),
        "requires_confirmation": defn.get("requires_confirmation", False),
        "depends_on": defn.get("depends_on", []),
    }
    tunnel = defn.get("tunnel")
    if tunnel:
        canonical["tunnel"] = {
            "base_url": tunnel.get("base_url", ""),
            "allowed_hosts": tunnel.get("allowed_hosts", []),
        }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _canonical_payload(definition_json: str, source_code: str) -> bytes:
    def_hash = hashlib.sha256(definition_json.encode()).digest()
    src_hash = hashlib.sha256(source_code.encode()).digest()
    return def_hash + src_hash


def sign_package(
    definition: dict,
    source_code: str,
    private_key_pem: str,
    publisher: str,
) -> PackageSignature:
    """Sign a tool package with the publisher's Ed25519 private key."""
    private_key = load_pem_private_key(private_key_pem.encode(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("key must be Ed25519")

    def_json = _canonical_definition(definition)
    payload = _canonical_payload(def_json, source_code)
    signature = private_key.sign(payload)

    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    return PackageSignature(
        publisher=publisher,
        public_key=public_pem,
        signature=b64encode(signature).decode(),
        payload_hash=hashlib.sha256(payload).hexdigest(),
    )


class KeyStore:
    """Manages the publisher's signing keys on disk.

    Keys are stored as PEM files in the keys directory:
        keys/{publisher}.key     — private key
        keys/{publisher}.pub     — public key
    """

    def __init__(self, keys_dir: str | Path = "keys"):
        self._dir = Path(keys_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(self, publisher: str) -> str:
        """Generate and store a keypair. Returns the public key PEM."""
        private_pem, public_pem = generate_keypair()
        (self._dir / f"{publisher}.key").write_text(private_pem)
        (self._dir / f"{publisher}.pub").write_text(public_pem)
        log.info("generated keypair for '%s'", publisher)
        return public_pem

    def get_private_key(self, publisher: str) -> str:
        path = self._dir / f"{publisher}.key"
        if not path.exists():
            raise FileNotFoundError(f"no private key for '{publisher}'")
        return path.read_text()

    def get_public_key(self, publisher: str) -> str:
        path = self._dir / f"{publisher}.pub"
        if not path.exists():
            raise FileNotFoundError(f"no public key for '{publisher}'")
        return path.read_text()

    def list_publishers(self) -> list[str]:
        return sorted({p.stem for p in self._dir.glob("*.pub")})

    def sign(self, publisher: str, definition: dict, source_code: str) -> PackageSignature:
        """Sign a package using a stored key."""
        private_key_pem = self.get_private_key(publisher)
        return sign_package(definition, source_code, private_key_pem, publisher)

# WireGuard tunnel: Tokyo ↔ Finland (ADR-010)

Step-by-step runbook for setting up the encrypted Finland → Tokyo
HTTP-over-/highfreq channel that supersedes the original "UFW-restrict
plaintext" approach in [ADR-009](../architecture.md#adr-009).

## Address plan

| Host | Public IP | WireGuard IP | Role |
|---|---|---|---|
| Tokyo (4VPS.su JP-cx21) | `147.45.49.40` | `10.99.0.1/24` | Server (`ListenPort = 51820`) |
| Finland (Hostkey) | `151.245.139.21` | `10.99.0.2/24` | Client (dial-out) |

## Prerequisites

- Tokyo VPS bootstrapped via `bootstrap_tokyo.sh` and reachable via SSH key auth
- Finland nginx already proxying `/highfreq` to Tokyo (via the public IP)
  — see `README.md` "Production layout" section

## One-time setup (≈10 min)

### 1. Install WireGuard on both hosts

```bash
ssh root@TOKYO   'apt-get install -y wireguard wireguard-tools'
ssh stailfx@FIN  'sudo apt-get install -y wireguard wireguard-tools'
```

### 2. Generate keys (privkey never leaves its host)

```bash
# Tokyo
ssh root@TOKYO 'umask 077 && mkdir -p /etc/wireguard
                wg genkey > /etc/wireguard/server-priv.key
                cat /etc/wireguard/server-priv.key | wg pubkey > /etc/wireguard/server-pub.key
                cat /etc/wireguard/server-pub.key'   # ← capture as $TOKYO_PUB

# Finland
ssh stailfx@FIN 'sudo mkdir -p /etc/wireguard && sudo chmod 700 /etc/wireguard
                 sudo bash -c "umask 077 && wg genkey > /etc/wireguard/client-priv.key &&
                               cat /etc/wireguard/client-priv.key | wg pubkey > /etc/wireguard/client-pub.key"
                 sudo cat /etc/wireguard/client-pub.key' # ← capture as $FIN_PUB
```

### 3. Write `/etc/wireguard/wg0.conf`

**Tokyo** (substitute the captured `$FIN_PUB`):

```ini
[Interface]
Address    = 10.99.0.1/24
ListenPort = 51820
PrivateKey = $(cat /etc/wireguard/server-priv.key)

[Peer]
PublicKey           = <FIN_PUB>
AllowedIPs          = 10.99.0.2/32
PersistentKeepalive = 25
```

**Finland** (substitute `$TOKYO_PUB`):

```ini
[Interface]
Address    = 10.99.0.2/24
PrivateKey = $(cat /etc/wireguard/client-priv.key)

[Peer]
PublicKey           = <TOKYO_PUB>
Endpoint            = 147.45.49.40:51820
AllowedIPs          = 10.99.0.1/32
PersistentKeepalive = 25
```

`chmod 600 /etc/wireguard/wg0.conf` on both.

### 4. UFW: open UDP 51820 on Tokyo from Finland's public IP only

```bash
ssh root@TOKYO 'ufw allow from 151.245.139.21 to any port 51820 proto udp \
                  comment "wireguard from Finland"'
```

### 5. Bring up the tunnel + persist on boot

```bash
ssh root@TOKYO   'systemctl enable --now wg-quick@wg0'
ssh stailfx@FIN  'sudo systemctl enable --now wg-quick@wg0'
```

### 6. Verify with ping

```bash
ssh stailfx@FIN  'ping -c 3 10.99.0.1'    # Finland → Tokyo
ssh root@TOKYO   'ping -c 3 10.99.0.2'    # Tokyo → Finland
ssh root@TOKYO   'wg show wg0'            # latest handshake should be < 30 s ago
```

### 7. Move the slim FastAPI to the WG interface

The `neucast-highfreq-web.service` unit (this directory) is already
configured to bind to `10.99.0.1` and depend on `wg-quick@wg0.service`.
Apply it:

```bash
scp docs/highfreq/deploy/neucast-highfreq-web.service \
    root@TOKYO:/etc/systemd/system/neucast-highfreq-web.service
ssh root@TOKYO 'systemctl daemon-reload && systemctl restart neucast-highfreq-web.service'
```

Verify uvicorn is bound only to wg0:

```bash
ssh root@TOKYO 'ss -tlnp | grep 8000'
# Expected: LISTEN  10.99.0.1:8000  ...
```

### 8. Tighten Tokyo UFW: replace the public-IP rule with a WG-IP rule

```bash
ssh root@TOKYO '
  ufw allow from 10.99.0.2 to any port 8000 proto tcp \
    comment "neucast-highfreq-web from Finland over WG"
  ufw delete allow from 151.245.139.21 to any port 8000 proto tcp
  ufw status verbose
'
```

After this step, the Tokyo public interface (`ens3`) accepts only:
- `22/tcp` (SSH, key-only auth)
- `51820/udp` from Finland's public IP (WG handshake)

Port `8000/tcp` is gone from the public surface entirely.

### 9. Switch Finland nginx upstream

Edit `/etc/nginx/sites-enabled/neucast.conf`:

```nginx
upstream neucast_tokyo {
    server 10.99.0.1:8000;   # via WireGuard, encrypted
    keepalive 8;
}
```

Then `sudo nginx -t && sudo systemctl reload nginx`.

### 10. End-to-end verify

```bash
# Public endpoint should still serve normally:
curl -s -o /dev/null -w "%{http_code}\n" https://neucast.ru/highfreq

# Sniff Tokyo's public interface for plaintext 8000 — must be ZERO packets:
ssh root@TOKYO 'timeout 10 tcpdump -i ens3 -nn -c 100 \
                  "tcp port 8000 and not net 10.99.0.0/24"' &
# In parallel, generate traffic:
for i in {1..5}; do curl -s https://neucast.ru/api/highfreq/status > /dev/null; done
# Expected tcpdump output: "0 packets captured"
```

## Operational notes

- **Reconnect behavior**: `PersistentKeepalive = 25` keeps the NAT/conntrack
  entry warm; re-handshake happens every 2 minutes automatically. If the
  Tokyo VPS reboots, the Finland side reconnects within ~30 s.
- **Revocation**: to rotate keys, re-run §2 + §3 + §5 (no need to touch UFW/nginx).
- **Adding a peer**: another `[Peer]` block in `/etc/wireguard/wg0.conf` on
  Tokyo + a separate `wg0.conf` on the new host. Address it as `10.99.0.N`,
  add the corresponding pubkey on Tokyo's side.
- **Debugging**: `journalctl -u wg-quick@wg0` on either host;
  `wg show wg0` for handshake state and transfer counters.

# LSEG App Key Rotation Guide

## When to rotate

Rotate the LSEG app key every **90 days** or immediately if you suspect a leak. The scheduler container will log a reminder when a rotation is due.

## Step-by-step

### 1. Generate a new key in the LSEG portal

1. Go to [LSEG Developer Portal](https://developers.lseg.com/en/api-catalog)
2. Sign in with your LSEG/Refinitiv account
3. Navigate to **My Account → App Keys** (or use the App Key Generator tool)
4. Either:
   - **Regenerate** your existing key (invalidates the old one immediately), or
   - **Create a new** app key (old key stays valid until you delete it — safer for zero-downtime rotation)
5. Copy the new app key string

### 2. Run the rotation script

From your deployment server (Hetzner):

```bash
cd ~/aris-macro-overlay
./docker/scripts/rotate-lseg-key.sh YOUR_NEW_APP_KEY_HERE
```

The script will:
1. Validate the key by attempting an LSEG API session
2. Back up your current `.env`
3. Swap `LSEG_APP_KEY` in `.env`
4. Restart `aris` and `scheduler` containers (Gateway untouched)
5. Record the rotation in `state/lseg_key_rotation.json`

### 3. Verify

```bash
docker compose logs aris --tail 50
```

Look for `"prices": "fresh"` in the pipeline health output. If you see `"prices": "missing"` or LSEG connection errors, the key may be invalid — restore from backup:

```bash
cp .env.bak.YYYYMMDD_HHMMSS .env
docker compose restart aris scheduler
```

### 4. Clean up (if you created a new key instead of regenerating)

Go back to the LSEG portal and delete the old app key.

## Rotation history

Check `state/lseg_key_rotation.json` for a log of all past rotations with timestamps and validation results.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FAIL: 401 Unauthorized` | Key invalid or account issue | Check key in LSEG portal, ensure account is active |
| `FAIL: connection refused` | LSEG API down or network issue | Retry later, check status.refinitiv.com |
| `SKIP: lseg_not_installed` | Running script locally without lseg-data | Safe to proceed — will validate on next pipeline run |
| Pipeline shows `prices: missing` after rotation | Key not propagated to container | Run `docker compose restart aris scheduler` |

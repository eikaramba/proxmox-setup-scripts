#!/bin/bash
echo ">>> USER SCRIPT: APPLYING ISSUE #107 WORKAROUND <<<"

# 1. Stop Supervisor from managing udev (prevents the "FATAL" crash log and Xorg hang)
sed -i 's|^autostart.*=.*$|autostart=false|' /etc/supervisor.d/udev.ini

# 2. PERMISSIONS: Fix /dev/uinput and /dev/uhid
# We attempt to fix them here, but if this fails, the Host Udev rule is required.
chmod 0666 /dev/uinput 2>/dev/null || echo " - Failed to chmod uinput (Host fix required)"
chmod 0666 /dev/uhid 2>/dev/null || echo " - Failed to chmod uhid (Host fix required)"

# 3. THE FIX from Issue #107
# Manually start udevd as a daemon. 
# This allows Sunshine to "talk" to the input system without Supervisor checking on it.
echo " - Starting systemd-udevd manually..."
/lib/systemd/systemd-udevd --daemon

# 4. Ensure default user is in the right groups
usermod -aG input,video,render default
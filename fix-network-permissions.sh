#!/bin/bash
# Fix NetworkManager permissions for Prismatron user

echo "Installing NetworkManager PolicyKit rule for user 'mark'..."

# Create rules.d directory if it doesn't exist
mkdir -p /etc/polkit-1/rules.d

# Create PolicyKit rule to allow mark user to manage NetworkManager
cat > /etc/polkit-1/rules.d/10-prismatron-networkmanager.rules << 'EOF'
/* Allow prismatron user to manage NetworkManager */
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.freedesktop.NetworkManager.") == 0 &&
        subject.user == "mark") {
        return polkit.Result.YES;
    }
});
EOF

echo "Adding mark user to netdev group..."
usermod -a -G netdev mark

echo "Restarting polkit service..."
systemctl restart polkit

echo "NetworkManager permissions fixed. Testing WiFi scan..."
su - mark -c "nmcli device wifi rescan"

echo "Done! The web interface should now be able to scan WiFi networks."

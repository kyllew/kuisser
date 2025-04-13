#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Node.js and npm (using Node.js 18.x LTS)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Git
sudo apt install -y git

# Install Nginx
sudo apt install -y nginx

# Install PM2 globally
sudo npm install -y pm2 -g

# Create directory for the application
sudo mkdir -p /var/www/quiz-app
sudo chown -R $USER:$USER /var/www/quiz-app

# Configure Nginx
sudo tee /etc/nginx/sites-available/quiz-app <<EOF
server {
    listen 80;
    server_name _;  # Replace with your domain name if you have one

    root /var/www/quiz-app/build;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

# Enable the Nginx configuration
sudo ln -s /etc/nginx/sites-available/quiz-app /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Print versions for verification
echo "Node.js version: $(node -v)"
echo "npm version: $(npm -v)"
echo "Git version: $(git --version)"
echo "PM2 version: $(pm2 -v)"

echo "Setup complete! You can now deploy your application." 
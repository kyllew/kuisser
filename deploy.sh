#!/bin/bash

# Exit on any error
set -e

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

# Configuration
S3_BUCKET="kuisser"  # Replace with your S3 bucket name
APP_DIR="/var/www/kuisser"
TEMP_DIR="/tmp/kuisser"
SERVER_DIR="/var/www/kuisser/server"

# Create temporary directory
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Download files from S3
echo "Downloading files from S3..."
aws s3 cp s3://$S3_BUCKET/kuisser.zip .

# Unzip the files
echo "Extracting files..."
unzip -q kuisser.zip
rm kuisser.zip

# Setup Node.js environment
echo "Setting up Node.js environment..."

# Install Node.js directly using package manager
echo "Installing Node.js 18.x..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify Node.js installation
node --version
npm --version

# Install global dependencies
echo "Installing global dependencies..."
sudo npm install -g typescript
sudo npm install -g vite
sudo npm install -g @cloudscape-design/components

# Clean install project dependencies
echo "Installing project dependencies..."
rm -rf node_modules
npm ci

# Fix permissions
echo "Setting correct permissions..."
sudo chown -R $USER:$USER .

# Create production build
echo "Creating production build..."
export PATH="$PATH:./node_modules/.bin"
npm run build

# Verify dist directory exists (Vite's output directory)
if [ ! -d "dist" ]; then
    echo "Error: Build directory (dist) not created. Check the build logs above."
    exit 1
fi

# Create server directory and setup server files
echo "Setting up server..."
mkdir -p $SERVER_DIR

# Copy server files
cp server/server.js $SERVER_DIR/
cp server/package.json $SERVER_DIR/

# Create server package.json
cat > $SERVER_DIR/package.json <<EOF
{
  "name": "kuisser-server",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "dotenv": "^16.0.3",
    "express": "^4.18.2"
  }
}
EOF

# Install server dependencies
echo "Installing server dependencies..."
cd $SERVER_DIR
npm install
cd $TEMP_DIR

# Copy build files to application directory
echo "Copying files to application directory..."
sudo mkdir -p $APP_DIR
sudo cp -r dist $APP_DIR/
sudo cp -r public $APP_DIR/
sudo cp -r server $APP_DIR/

# Setup PM2
echo "Setting up PM2..."
sudo npm install -g pm2

# Create PM2 ecosystem file
cat > $APP_DIR/ecosystem.config.js <<EOF
module.exports = {
  apps: [{
    name: 'kuisser',
    script: 'server/server.js',
    cwd: '$APP_DIR',
    env: {
      NODE_ENV: 'production',
      PORT: 5000
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
};
EOF

# Start application with PM2
echo "Starting application with PM2..."
cd $APP_DIR
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Setup Nginx
echo "Setting up Nginx..."
sudo apt-get install -y nginx

# Create Nginx configuration
cat > /etc/nginx/sites-available/kuisser <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable Nginx configuration
sudo ln -sf /etc/nginx/sites-available/kuisser /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Cleanup
echo "Cleaning up..."
cd /
sudo rm -rf $TEMP_DIR

echo "Deployment completed successfully!"
echo "Application is running at http://localhost"
echo "PM2 process list:"
pm2 list 
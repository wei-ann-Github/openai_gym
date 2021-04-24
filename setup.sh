# Reference: https://faun.pub/deploy-flask-app-with-nginx-using-gunicorn-7fda4f50066a

# Symlink app.service in /etc/systemd/system/
PROJECT=$(basename "$PWD")

sudo cp $(pwd)/app.service /etc/systemd/system/${PROJECT}_app.service

# Start the Gunicorn service we created and enable it so that it starts at boot:
sudo systemctl start app
sudo systemctl enable app

# Move nginx.conf into /etc/nginx/sites-available/
sudo ln -s $(pwd)/nginx.conf /etc/nginx/sites_enabled/${PROJECT}.conf

sudo systemctl restart nginx

# If neccessary, adjust our firewall to allow access to the Nginx server:
sudo ufw allow 'Nginx Full'

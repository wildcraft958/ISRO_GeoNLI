FROM node:20-bullseye-slim

WORKDIR /app

# Install OpenSSL for crypto support
RUN apt-get update && \
   apt-get install -y --no-install-recommends \
   openssl \
   && rm -rf /var/lib/apt/lists/*

# Copy package files for dependency installation
COPY package.json package-lock.json* ./

# Use a more efficient installation strategy
RUN npm config set registry https://registry.npmjs.org/ && \
   npm install --legacy-peer-deps --no-fund --no-audit --loglevel=error

# Copy the rest of the application code
COPY . .

EXPOSE 5173

# Use a simplified start command that should work consistently
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
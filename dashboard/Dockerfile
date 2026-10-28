FROM node:18-slim

WORKDIR /app
RUN npm install -g serve

COPY index.html .

EXPOSE 3000
CMD ["serve", "-s", ".", "-l", "3000"]

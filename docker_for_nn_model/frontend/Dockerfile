
FROM node:18-alpine AS build

WORKDIR /app

# Копируем package файлы
COPY package*.json ./

# Устанавливаем зависимости
RUN npm ci --only=production && \
    npm cache clean --force

# Копируем исходный код
COPY . .

# Собираем приложение
RUN npm run build


FROM nginx:1.25-alpine AS production

# Копируем nginx конфигурацию
COPY nginx.conf /etc/nginx/nginx.conf

# Копируем собранное приложение из build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Удаляем дефолтную страницу nginx
RUN rm /etc/nginx/conf.d/default.conf

# Создаем non-root пользователя
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Меняем владельца файлов
RUN chown -R nodejs:nodejs /usr/share/nginx/html && \
    chown -R nodejs:nodejs /var/cache/nginx && \
    chown -R nodejs:nodejs /var/log/nginx && \
    chown -R nodejs:nodejs /etc/nginx/conf.d

RUN touch /var/run/nginx.pid && \
    chown -R nodejs:nodejs /var/run/nginx.pid

USER nodejs

# Экспонируем порт
EXPOSE 8080

# Запускаем nginx
CMD ["nginx", "-g", "daemon off;"]
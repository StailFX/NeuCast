# Общие proxy-заголовки, включаемые во все location-блоки
proxy_http_version 1.1;
proxy_set_header Host              $host;
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header Connection        "";

proxy_connect_timeout 10s;
proxy_send_timeout    180s;
proxy_read_timeout    180s;

# Буферизация ответов (снижает нагрузку на upstream)
proxy_buffering on;
proxy_buffers 16 16k;
proxy_busy_buffers_size 32k;

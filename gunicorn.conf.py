import os
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
worker_class = "gthread"
workers = 1
threads = 8
timeout = 300

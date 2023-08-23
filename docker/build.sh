#/bin/bash
docker build --build-arg HTTP_PROXY=$http_proxy \
--build-arg HTTPS_PROXY=$http_proxy --build-arg NO_PROXY="$no_proxy" \
--build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy \
--build-arg no_proxy="$no_proxy" .. -t cmrxrecon-jabber -f Dockerfile

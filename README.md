# reporter-quality-testing-rig
OTv2: Docker containers and Jupyter Notebooks for testing Open Traffic Reporter quality using synthesized GPS data


### Setting up test environment
1. Clone Valhalla docker [repo](https://github.com/valhalla/docker)
2. Build Valhalla docker image from source with:
    - `./build.sh source latest`
    - ensure port 8003 is exposed by **Dockerfile-source**
    - ensure `ENV_PRIME_LISTEN` gets set to 127.0.0.1:8003:
        - add `--httpd-service-listen tcp://127.0.0.1:8003` param to `valhalla build config` call in **Dockerfile-source**
3. Clone Open Traffic reporter [repo](https://github.com/opentraffic/reporter.git)
4. Build Open Traffic reporter image with `docker build --tag opentraffic/reporter:latest --force-rm .`
5. export env vars:
    - `export VALHALLA_DOCKER_DATAPATH=</path/to/valhalla/tiles.tar>`
    - `export DATAPATH=</path/to/opentraffic/reporter/tiles.tar>`
6. `docker-compose up`





### TODO
- containerize test environment
- enumerate params to tune:
    - sigma-z
    - beta
    - sampling rate
- expose params as top-level container args

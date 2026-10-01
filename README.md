# VoterX

VoterX is a small Node.js/Express service that demonstrates Redis-backed sessions and cache-aside user loading. The repository also contains a Docker image definition and Kubernetes manifests for a Redis-backed deployment, with Traefik as an optional ingress proxy.

## Requirements

- Node.js 20 or newer and npm.
- Redis for running the app with its default stores.
- Docker Engine/Desktop to build and run the container.
- `kubectl` with access to a cluster to deploy the Kubernetes manifests.

## Install and run locally

```sh
npm ci
npm run build
npm test
npm start
```

The API listens on port `3000` by default. Set `PORT` to change it. `npm run dev` starts the app with Node's file watcher. The app loads `.env` through `dotenv`; do not commit secrets from that file.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `3000` | HTTP listening port |
| `REDIS_URL` | `redis://127.0.0.1:6379` | Redis connection URL for sessions and cache |
| `NODE_ENV` | unset | When set to `production`, session cookies include `Secure` |

Both the app's session store and cache use the same Redis URL and distinct Keyv namespaces. Redis must be reachable from the app process.

## HTTP API

### `GET /healthz`

Returns `{"status":"ok"}`. This endpoint checks that the HTTP process responds; it does not check Redis availability.

### `POST /session`

Accepts a JSON request body and calls the `authenticateUser` function injected into `createApp`. A successful authenticator returns a user with a positive, safe-integer `id`; the handler stores a session for seven days, sets an `HttpOnly; SameSite=Lax` cookie, and responds with `204 No Content`. In production, the cookie also has the `Secure` flag.

The default authenticator rejects all credentials. Wire it to a real authentication system before expecting this route to issue sessions. The function receives `(request.body, request)` and can be provided when constructing the app with `createApp({ authenticateUser })`.

### `GET /user`

Requires the `sessionId` cookie and a non-expired session in Redis. Returns the user object loaded by the injected `userLoader`; successful values are cached for five minutes. Missing/invalid sessions return `401`.

The default user loader returns placeholder profile data. Provide a real `userLoader` when constructing the app with `createApp({ userLoader })`.

## Cache behavior

`createCacheAside(store, defaultTtl, onCacheError)` provides the cache-aside helper used by the user endpoint:

- Reads cached data first; on a miss it invokes the loader, writes the result with a TTL, and returns it.
- Coalesces concurrent misses for the same key within one Node.js process.
- Reports cache read/write errors and still tries the loader, so Redis cache outages need not block a loaded response.
- Propagates loader failures and does not cache failed results.

The in-flight request map is per process, not shared across replicas. Redis is shared by replicas, but simultaneous misses arriving at different pods can still invoke the loader more than once.

## Tests and build check

```sh
npm run build
npm test
```

The build script performs JavaScript syntax checks; it is not a transpilation step. The Node test runner tests the HTTP session flow and cache behavior, including TTL expiration, concurrent misses, and cache failures. Docker image builds run both commands in the build stage.

## Docker

Build the multi-stage image (the build stage runs syntax checks and unit tests):

```sh
docker build -t voterx:1.0.0 .
```

Run the API on the host network and configure the Redis URL for the environment where the container runs:

```sh
docker run --rm -p 3000:3000 -e REDIS_URL=redis://host.docker.internal:6379 voterx:1.0.0
```

The Dockerfile uses a Node 24 build/runtime image, installs production dependencies in the final image, excludes development files through `.dockerignore`, and runs as the non-root `node` user. For a container-to-container Redis setup, place both services on the same Docker network and use the Redis service name in `REDIS_URL`.

`compose.yaml` currently defines Redis only; it does not start the API. It persists Redis data in a named volume and publishes Redis on host port `6379`.

## Kubernetes

The Kustomize configuration under `k8s/` is the deployment entry point:

```sh
kubectl apply -k k8s
kubectl -n voterx rollout status deployment/redis
kubectl -n voterx rollout status deployment/traefik
kubectl -n voterx rollout status deployment/voterx-api
kubectl -n voterx get service traefik-private --watch
```

The manifests create the `voterx` namespace, a Redis service, a three-replica API Deployment and ClusterIP Service, and a two-replica Traefik proxy exposed through an AWS internal Network Load Balancer. The API Ingress is routed through Traefik. The API container image is `chitnisa/voterx:1.0.0`; make that image available to the cluster before applying the manifests.

This manifest set is aimed at EKS with the AWS Load Balancer Controller installed. The `traefik-private` Service uses AWS-specific annotations and the `service.k8s.aws/nlb` load-balancer class. The NLB is internal; accessing it and the API requires network access from the appropriate VPC. Detailed API Gateway/VPC Link prerequisites and setup are in [`kubernetes-deploy.md`](kubernetes-deploy.md) and [`api-gateway-vpc-link.md`](api-gateway-vpc-link.md).

The Kubernetes Redis manifest runs one Redis replica with persistence disabled. Redis restarts can therefore invalidate sessions and cache entries. The deployment does not configure Redis authentication or TLS; use appropriate network controls and production Redis hardening for environments that require them.

## Repository map

- `app.js` — Express routes, session handling, app factory, and HTTP server entry point.
- `cache.js` — Redis store creation and cache-aside helper.
- `app.test.js`, `cache.test.js` — Node built-in test suite.
- `Dockerfile`, `.dockerignore` — multi-stage application image build.
- `compose.yaml` — local Redis service.
- `k8s/` — namespace, Redis, API, Traefik, and Kustomize manifests.
- `kubernetes-deploy.md` — image publishing/loading and EKS deployment steps.
- `api-gateway-vpc-link.md` — private API Gateway integration overview.
- `api-gateway-resource-policy.template.json` — API Gateway client-IP allow-list policy template; replace `${ALLOWED_CLIENT_CIDR}` before use.
- `.gitignore` — excludes local environment files and installed dependencies.

## Important limitations

- Authentication is an injected interface only; there is no built-in credential store, password verification, or user database.
- The default profile loader is sample data, not a database lookup.
- There is no logout/session-revocation route; sessions expire through their seven-day Redis TTL.
- `/healthz` is a process health check, not a Redis readiness check.
- Cache miss coalescing only applies within an individual API process.
- The Kubernetes Redis deployment is ephemeral and intended as a simple starter setup, not a highly available production data store.

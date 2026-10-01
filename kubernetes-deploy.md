# EKS deployment

This deployment uses an AWS internal Network Load Balancer, so run it on EKS with the AWS Load Balancer Controller installed. The request path is:

`Client -> API Gateway -> VPC Link -> internal NLB -> Traefik -> voterx-api -> Redis`

## Prerequisites

- An EKS cluster and `kubectl` access to it.
- AWS Load Balancer Controller installed in the cluster.
- Docker Desktop or Docker Engine, plus Docker Hub access for `chitnisa/voterx`.
- An API Gateway VPC Link in private subnets that can reach the NLB.
- An NLB security group that permits only the VPC Link security group. Without this, a client inside the VPC could bypass API Gateway by connecting to the internal NLB directly.

## Deploy

1. Make `chitnisa/voterx:1.0.0` available to the cluster.

For EKS or any remote cluster, push it to Docker Hub:

```sh
docker build -t chitnisa/voterx:1.0.0 .
docker push chitnisa/voterx:1.0.0
```

For a local cluster, load the included archive instead. Do not push unless you want the image available outside your machine:

```sh
# Docker Desktop Kubernetes
docker load -i voterx.tar

# kind
kind load image-archive voterx.tar --name YOUR_CLUSTER_NAME

# minikube
minikube image load voterx.tar
```

`k8s/api-k8s.yaml` uses `imagePullPolicy: IfNotPresent`, so Kubernetes uses the loaded image instead of attempting a registry pull.

2. Replace `${ALLOWED_CLIENT_CIDR}` in `api-gateway-resource-policy.template.json` with the client CIDR allowed to invoke API Gateway.
3. Apply the complete stack:

```sh
kubectl apply -k k8s
kubectl -n voterx rollout status deployment/redis
kubectl -n voterx rollout status deployment/traefik
kubectl -n voterx rollout status deployment/voterx-api
kubectl -n voterx get service traefik-private --watch
```

4. Use the listener ARN from the `traefik-private` NLB to create the API Gateway HTTP proxy private integration through the VPC Link.

The `voterx-api` Service remains `ClusterIP`; only Traefik is NLB-facing. Traefik watches the Ingress, Service, and EndpointSlices, so it automatically uses healthy API pods as replicas change.

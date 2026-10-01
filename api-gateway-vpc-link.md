# API Gateway private integration

Traffic must flow in this order:

`Client -> API Gateway IP policy -> VPC Link -> internal NLB -> Traefik -> voterx-api pods`

The NLB is internal and targets Traefik pod IPs. It is an API Gateway integration target; it cannot make routing decisions from the original client IP. Traefik supplies reverse-proxy routing, load balancing, and Kubernetes service discovery for `voterx-api`. Apply client IP allow/deny rules to API Gateway with `api-gateway-resource-policy.template.json` (replace `${ALLOWED_CLIENT_CIDR}` before applying it). Use AWS WAF when the policy needs a large or frequently updated IP set.

Prerequisites:

- An EKS cluster with the AWS Load Balancer Controller installed.
- A Kubernetes cluster version supported by Traefik v3.7.
- The container image URI substituted in `api-k8s.yaml`.
- A Redis service deployed from `redis-k8s.yaml`.
- An API Gateway VPC link in private subnets that can reach the NLB.

Deploy the workload and wait for the NLB hostname. The EKS-specific runbook is in `kubernetes-deploy.md`.

```sh
kubectl apply -k k8s
kubectl -n voterx get service traefik-private --watch
```

Create an API Gateway private HTTP proxy integration using the NLB listener ARN and the VPC link ID. The API Gateway, VPC link, and NLB must be in the same AWS account. Configure API Gateway to forward the request path without its stage prefix.

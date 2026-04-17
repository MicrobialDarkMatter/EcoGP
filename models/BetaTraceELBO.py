from pyro.infer import Trace_ELBO
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item


class BetaTraceELBO(Trace_ELBO):
    def __init__(self, beta=1.0, *args, **kwargs):
        """
        :param beta: Scaling factor for the KL divergence term (default=1.0, standard ELBO)
        """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def loss(self, model, guide, *args, **kwargs):
        """
        Computes the beta-ELBO loss.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - self.beta * torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0.0
        surrogate_elbo_particle = 0.0

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle += torch_item(site["log_prob_sum"])
                surrogate_elbo_particle += site["log_prob_sum"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle -= self.beta * torch_item(site["log_prob_sum"])
                surrogate_elbo_particle -= self.beta * site["log_prob_sum"]

        return -elbo_particle, -surrogate_elbo_particle

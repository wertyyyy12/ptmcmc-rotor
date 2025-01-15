class BayesianModel:
    def generate_data(self, num_data_points):
        pass

    def model(self, data_input, parameters):
        pass

    def lnlikelihood(self, data_input, data_output, parameters):
        pass

    def lnprior(self, parameters):
        pass

    def _unnormalized_posterior(self, parameters, data_input, data_output, is_ptmcmc):
        pass

    def unnormalized_posterior(self, parameters, is_ptmcmc):
        pass
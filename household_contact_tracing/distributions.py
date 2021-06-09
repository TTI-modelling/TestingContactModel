# Maths that describes statistical distributions

import math

from scipy import integrate as si

gen_shape = 2.826
gen_scale = 5.665


def weibull_pdf(t):
    out = (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1) * math.exp(-(t / gen_scale)**gen_shape)
    return out


def weibull_survival(t):
    return math.exp(-(t / gen_scale)**gen_shape)


def unconditional_hazard_rate(t, survive_forever):
    """
    Borrowed from survival analysis.

    To get the correct generation time distribution, set the probability
    of a contact on day t equal to the generation time distribution's hazard rate on day t

    Since it is not guaranteed that an individual will be infected, we use improper variables and rescale appropriately.
    The R0 scaling parameter controls this, as R0 is closely related to the probability of not being infected
    The relationship does not hold exactly in the household model, hence model tuning is required.

    Notes on the conditional variable stuff https://data.princeton.edu/wws509/notes/c7s1

    Returns
    The probability that a contact made on day t causes an infection.

    Notes:
    Currently this is using a weibull distribution, as an example.
    """
    unconditional_pdf = (1 - survive_forever) * weibull_pdf(t)
    unconditional_survival = (1 - survive_forever) * weibull_survival(t) + survive_forever
    return unconditional_pdf / unconditional_survival


def current_hazard_rate(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), 0, 0.5)[0]
    else:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t - 0.5, t + 0.5)[0]


def current_rate_infection(t):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: weibull_pdf(t), 0, 0.5)[0]
    else:
        return si.quad(lambda t: weibull_pdf(t), t - 0.5, t + 0.5)[0]


def negbin_pdf(x, m, a):
    """
    We need to draw values from an overdispersed negative binomial distribution, with non-integer inputs. Had to
    generate the numbers myself in order to do this.
    This is the generalized negbin used in glm models I think.

    m = mean
    a = overdispertion
    """
    A = math.gamma(x + 1 / a) / (math.gamma(x + 1) * math.gamma(1 / a))
    B = (1 / (1 + a * m))**(1 / a)
    C = (a * m / (1 + a * m))**x
    return A * B * C


def compute_negbin_cdf(mean, overdispersion, length_out=100):
    """
    Computes the overdispersed negative binomial cdf, which we use to generate random numbers by generating uniform(0,1)
    rv's.
    """
    pdf = [negbin_pdf(i, mean, overdispersion) for i in range(length_out)]
    cdf = [sum(pdf[:i]) for i in range(length_out)]
    return cdf
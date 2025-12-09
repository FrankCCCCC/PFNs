import torch

from .prior import Batch


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    get_batch,
    device="cpu",
    hyperparameters=None,
    **kwargs,
):
    """
    This is not part of the paper, but feel welcome to use this to write a better version of our user prior.


    This function assumes that every x is in the range [0, 1].
    Style shape is (batch_size, 3*num_features) under the assumption that get_batch returns a batch
    with shape (seq_len, batch_size, num_features).
    The style is build the following way: [prob_of_feature_1_in_range, range_min_of_feature_1, range_max_of_feature_1, ...]



    :param batch_size:
    :param seq_len:
    :param num_features:
    :param get_batch:
    :param device:
    :param hyperparameters:
    :param kwargs:
    :return:
    """

    if hyperparameters is None:
        hyperparameters = {}

    maximize = hyperparameters.get("condition_on_area_maximize", True)
    size_range = hyperparameters.get("condition_on_area_size_range", (0.2, 0.99))
    distribution = hyperparameters.get("condition_on_area_distribution", "uniform")
    assert distribution in ["uniform"]
    extra_samples = hyperparameters.get("condition_on_area_extra_samples", 0)

    batch: Batch = get_batch(
        batch_size=batch_size,
        seq_len=seq_len + extra_samples,
        num_features=num_features,
        device=device,
        hyperparameters=hyperparameters,
        **kwargs,
    )
    assert batch.style is None

    d = batch.x.shape[2]

    division_size = (
        torch.rand(batch_size, d, device=device) * (size_range[1] - size_range[0])
        + size_range[0]
    )
    division_start = torch.rand(batch_size, d, device=device) * (1 - division_size)

    assert batch.target_y.shape[2] == 1, "Only support single objective."

    optima_inds = (
        batch.target_y.argmax(1).squeeze(-1)
        if maximize
        else batch.target_y.argmin(0).squeeze(-1)
    )  # batch_size, d

    optima = batch.x[torch.arange(batch_size), optima_inds]  # batch_size, d

    is_inside = (division_start <= optima) & (
        optima <= division_start + division_size
    )  # batch_size, d

    # hint_probs = torch.rand(batch_size, d, device=device) # probs are chosen randomly
    # hint probs need to be drawn dependent on whether it is inside or not
    # what we want is R ~ Uniform(0, 1), and we now sample p(R=r|Ber(R)=1) and p(R=r|Ber(R)=0)
    # that is: p(R=r|Ber(R)=1) = r / 0.5 = 2.0 * r, and p(R=r|Ber(R)=0) = (1-r) / 0.5 = 2.0 * (1-r)
    # we can compute the icdfs as icdf(|Ber(R)=1) = np.sqrt(u), icdf(|Ber(R)=0)= 1 - np.sqrt(1 - u)

    hint_probs = torch.where(
        is_inside,
        torch.sqrt(torch.rand(batch_size, d, device=device)),
        1 - torch.sqrt(1 - torch.rand(batch_size, d, device=device)),
    )

    batch.style = torch.stack(
        [hint_probs, division_start, division_start + division_size], 2
    )  # batch_size, d, 3

    skip_style_prob = hyperparameters.get("condition_on_opt_area_skip_style_prob", 0.0)

    skip_style_mask = torch.rand(batch_size, device=device) < skip_style_prob

    # set to nan for the encoder to figure this out
    batch.style[skip_style_mask, :, :] = torch.nan

    if extra_samples:
        batch.x = batch.x[:, :-extra_samples]
        batch.y = batch.y[:, :-extra_samples]
        batch.target_y = batch.target_y[:, :-extra_samples]

    return batch

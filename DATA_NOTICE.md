# Data notice

The product catalog is built from the **Amazon Berkeley Objects (ABO)** dataset,
licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
This project is non-commercial, and the dataset files are not stored in this repository;
the scripts download and process them on your own machine.

> Collins, J., Goel, S., Deng, K., Luthra, A., Xu, L., Gundogdu, E., Zhang, X.,
> Yago Vicente, T. F., Dideriksen, T., Arora, H., Guillaumin, M., and Malik, J.
> *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding.* CVPR 2022.
> https://arxiv.org/abs/2110.06199

Changes made to the data: listings are filtered to English-named furniture and
home decor product types, fields are flattened and converted to metric units,
and **prices are generated** (ABO has none). See `catalog/pricing.py` for how.

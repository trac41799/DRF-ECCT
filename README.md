# DRF-ECCT
This is the original repository for the article:
N. D. Trac and K. Sunghwan, “DRF-ECCT: Dynamic reliability filter for error correction code
transformer,” in Procs Int. Conf. Green Human Inf. Tech. (ICGHIT), Feb. 2024, pp. 18-23.

The full paper can be read by accessing [this file](DRF-ECCT/paper/ICGHIT2024_Proceeding_CI5_NgDangTrac.pdf).

This article was awarded best paper by the committee of the ICGHIT2024 Conference. Proof of the best paper award:

[Certification of Best paper award](DRF-ECCT/paper/BestPaper_DRF-ECCT.pdf)

[Announcement email of best paper award](DRF-ECCT/paper/ICGHIT2024_Announcement_best_paper_award.pdf)

## Abstract
In this work, we propose a dynamic reliability filter (DRF) mechanism to enhance the code-aware self-attention of error correction code transformers (ECCT). The DRF is designed to dynamically filter out the low-magnitude positions in the query and key matrices, amplifying the reliable information. By this mechanism, the code-aware self-attention of the ECCT is better utilized, effectively extracting the noise-corrupted positions from the channel output. Additionally, layer reuse is introduced for the feed-forward sub-layers to increase model efficiency, enabling the use of a deeper network with fewer parameters compared to the baseline ECCT. The experimental results show considerable improvements in bit error rates compared to the baseline ECCT, while also significantly increasing the convergence speed during the training stage.

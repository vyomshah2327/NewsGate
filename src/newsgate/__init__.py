"""NewsGate: when does financial news add predictive value beyond prices?

Package layout (each sub-package has one job):

* ``config``      typed, validated configuration loaded from YAML
* ``data``        news and price sources, cleaning, trading calendar, timestamp alignment
* ``sentiment``   interchangeable sentiment scorers (strategy pattern + registry)
* ``events``      causal clustering of headlines into events and novelty scores
* ``features``    feature blocks combined by a pipeline (price, attention, sentiment, event)
* ``labels``      forward-return targets and train-only large-move thresholds
* ``models``      probabilistic classifiers behind one interface (factory + registry)
* ``gate``        the incremental-value gate (use prices, use news, or abstain)
* ``evaluation``  purged expanding-window splits, metrics, block bootstrap, Holm correction
* ``viz``         the figures used in the report
* ``pipeline``    orchestration of the stages end to end
"""

__version__ = "0.1.0"

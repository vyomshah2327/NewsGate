"""Price features (prefix ``px``): the matched price-only information set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.features.base import FeatureBlock, FeatureContext, per_ticker


class PriceFeatures(FeatureBlock):
    prefix = "px"

    def _compute(self, ctx: FeatureContext) -> pd.DataFrame:
        cfg = ctx.cfg
        market = ctx.market.set_index("date")["mkt_logret"].sort_index()

        def one(g: pd.DataFrame) -> pd.DataFrame:
            logp = np.log(g["adj_close"])
            ret = logp.diff()
            out = pd.DataFrame(index=g.index)
            for k in cfg.return_lags:
                out[f"px_ret_{k}"] = ret.shift(k)
            for w in cfg.momentum_windows:
                out[f"px_mom_{w}"] = logp - logp.shift(w)
            for w in cfg.vol_windows:
                out[f"px_vol_{w}"] = ret.rolling(w).std()
            out["px_absret_0"] = ret.abs()
            out["px_volratio"] = g["volume"] / g["volume"].rolling(cfg.volume_window).mean()
            out["px_range"] = (g["high"] - g["low"]) / g["close"]
            mkt = g["date"].map(market)
            out["px_mkt_ret_0"] = mkt.to_numpy()
            out["px_rel_ret_0"] = (ret - mkt).to_numpy()
            weekday = g["date"].dt.dayofweek
            for d in range(5):
                out[f"px_dow_{d}"] = (weekday == d).astype("float64")
            return out

        frame = per_ticker(ctx.prices, one)
        mkt_vol = market.rolling(20).std().rename("px_mkt_vol_20")
        frame = frame.join(mkt_vol, on="date")
        return frame

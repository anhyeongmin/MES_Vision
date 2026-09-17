"""Independent model ownership for overview and detail capture domains."""
from copy import deepcopy
from mes_vision.training.data import require
from mes_vision.inspection.model_registry import default_registry
from mes_vision.operation.engine import Models
from .vision import StationVision


class StationModels:
    def __init__(self,root,overview_asset,detail_product,*,capture_domains,registry=None):
        require(capture_domains.get('overview_objects')=='overview','Overview training domain must be explicitly registered')
        for role in ('objects','defects','anomaly','geometry'):
            if detail_product.get(role):
                require(capture_domains.get(role)=='detail','Close-up data/criteria must be registered for '+role)
        self.registry=registry or default_registry()
        self.overview_asset=deepcopy(overview_asset); self.product=deepcopy(detail_product)
        self.overview=None; self.detail=Models(root,self.product,registry=self.registry); self.ready=False

    def load(self):
        require(not self.ready and self.overview is None,'Models are already loaded')
        try:
            self.overview=self.registry.create(self.overview_asset,'objects'); self.overview.load()
            self.detail.load(); self.ready=True
        except BaseException:
            self.close(); raise

    def service(self,*,overview_workspace,detail_workspace):
        require(self.ready,'Load scan models before creating the inspection service')
        return StationVision(self.overview.adapter,self.detail.detector,self.detail.inspectors,self.detail.policy,
            product_id=self.product['id'],overview_workspace=overview_workspace,detail_workspace=detail_workspace,
            detail_quality=self.product['quality'])

    def close(self):
        self.ready=False
        try: self.detail.close()
        finally:
            if self.overview: self.overview.close(); self.overview=None

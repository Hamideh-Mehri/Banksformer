

class FieldInfo:
    def __init__(self, n_tcode, CAT_FIELDS=['tcode_num'], CONT_FIELDS=['log_amount_sc'], DATE_FIELDS=['dow', 'month', "day", 'dtme', 'td_sc']):

        self.CAT_FIELDS = CAT_FIELDS
        self.CONT_FIELDS = CONT_FIELDS
        self.DATE_FIELDS = DATE_FIELDS
        self.n_tcode = n_tcode
        
        # cl - clock encoding (2d)
        # oh - One-hot encoding
        # raw - no encoding
        # cl-i -  clock integer: transforms [1, 2, ..., n] -> [1, 2, ..., n-1, 0]
        self.INP_ENCODINGS = {
            "day": "cl",
            "dtme": "cl",
            "dow": "cl",
            "month": "cl",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh"
        }
        self.TAR_ENCODINGS = {
            "day": "cl-i",
            "dtme": "cl-i",
            "dow": "cl-i",
            "month": "cl-i",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw"
        }

        
        self.DATA_KEY_ORDER = CAT_FIELDS + DATE_FIELDS + CONT_FIELDS
        self.LOSS_TYPES = self._get_loss_types()
        self.CLOCK_DIMS = self._get_clock_dims()
        self.FIELD_DIMS_IN, self.FIELD_DIMS_TAR, self.FIELD_DIMS_NET, self.FIELD_STARTS_IN, self.FIELD_STARTS_TAR, self.FIELD_STARTS_NET = self._get_field_dims_and_starts()
        
    def _get_loss_types(self):
        date_loss = "scce"         # 'scce': sparse categorical cross entropy``
        return {"day": date_loss,
                "dtme": date_loss,
                "dow": date_loss,
                "month": date_loss,
                "tcode_num": date_loss,
                "td_sc": "pdf",
                "log_amount_sc": "pdf"}

    def _get_clock_dims(self):
        return {"day": 31,
                "dtme": 31,
                "dow": 7,
                "month": 12}

    def _get_field_dims_and_starts(self):

        
        
        ENCODING_INP_DIMS_BY_TYPE = {'cl':2, 
                                'oh':None, 
                                'raw':1}

        ENCODING_TAR_DIMS_BY_TYPE = {'cl-i': 1, 
                                'raw': 1}
        
        FIELD_DIMS_IN  = {}
        FIELD_DIMS_TAR = {}
        FIELD_DIMS_NET = {}

        for k in self.DATA_KEY_ORDER:
            FIELD_DIMS_IN[k] = ENCODING_INP_DIMS_BY_TYPE[self.INP_ENCODINGS[k]]
            FIELD_DIMS_TAR[k] = ENCODING_TAR_DIMS_BY_TYPE[self.TAR_ENCODINGS[k]]

            if self.TAR_ENCODINGS[k] == "raw":
                FIELD_DIMS_NET[k] = 2
            elif self.TAR_ENCODINGS[k] == "cl-i":
                FIELD_DIMS_NET[k] = self.CLOCK_DIMS[k]
            else:
                raise Exception(f"Error getting network dim for field = {k}")
        for field in self.CAT_FIELDS:      
            FIELD_DIMS_IN[field] = self.n_tcode   
            FIELD_DIMS_NET[field] = self.n_tcode

        FIELD_STARTS_IN = self._compute_field_starts(FIELD_DIMS_IN)
        FIELD_STARTS_TAR = self._compute_field_starts(FIELD_DIMS_TAR)
        FIELD_STARTS_NET = self._compute_field_starts(FIELD_DIMS_NET)

        return FIELD_DIMS_IN, FIELD_DIMS_TAR, FIELD_DIMS_NET, FIELD_STARTS_IN, FIELD_STARTS_TAR, FIELD_STARTS_NET

    def _compute_field_starts(self, field_dims):
        field_starts = {}
        start = 0
        for k in self.DATA_KEY_ORDER:
            field_starts[k] = start
            start += field_dims[k]
        return field_starts





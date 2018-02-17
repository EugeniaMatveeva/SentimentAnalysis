from collections import namedtuple, defaultdict
import csv
import os


FIELDS = ("Entry, Source, Positiv, Negativ, Pstv, Affil, Ngtv, Hostile, Strong,"
          " Power, Weak, Submit, Active, Passive, Pleasur, Pain, Feel, Arousal,"
          " EMOT, Virtue, Vice, Ovrst, Undrst, Academ, Doctrin, Econ, Exch, "
          "ECON, Exprsv, Legal, Milit, Polit, POLIT, Relig, Role, COLL, Work, "
          "Ritual, SocRel, Race, Kin, MALE, Female, Nonadlt, HU, ANI, PLACE, "
          "Social, Region, Route, Aquatic, Land, Sky, Object, Tool, Food, "
          "Vehicle, BldgPt, ComnObj, NatObj, BodyPt, ComForm, COM, Say, Need, "
          "Goal, Try, Means, Persist, Complet, Fail, NatrPro, Begin, Vary, "
          "Increas, Decreas, Finish, Stay, Rise, Exert, Fetch, Travel, Fall, "
          "Think, Know, Causal, Ought, Perceiv, Compare, Eval, EVAL, Solve, "
          "Abs, ABS, Quality, Quan, NUMB, ORD, CARD, FREQ, DIST, Time, TIME, "
          "Space, POS, DIM, Rel, COLOR, Self, Our, You, Name, Yes, No, Negate, "
          "Intrj, IAV, DAV, SV, IPadj, IndAdj, PowGain, PowLoss, PowEnds, "
          "PowAren, PowCon, PowCoop, PowAuPt, PowPt, PowDoct, PowAuth, PowOth, "
          "PowTot, RcEthic, RcRelig, RcGain, RcLoss, RcEnds, RcTot, RspGain, "
          "RspLoss, RspOth, RspTot, AffGain, AffLoss, AffPt, AffOth, AffTot, "
          "WltPt, WltTran, WltOth, WltTot, WlbGain, WlbLoss, WlbPhys, WlbPsyc, "
          "WlbPt, WlbTot, EnlGain, EnlLoss, EnlEnds, EnlPt, EnlOth, EnlTot, "
          "SklAsth, SklPt, SklOth, SklTot, TrnGain, TrnLoss, TranLw, MeansLw, "
          "EndsLw, ArenaLw, PtLw, Nation, Anomie, NegAff, PosAff, SureLw, If, "
          "NotLw, TimeSpc, FormLw, Othtags, Defined")

InquirerLexEntry = namedtuple("InquirerLexEntry", FIELDS)
FIELDS = InquirerLexEntry._fields


class InquirerLexTransform():
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ".split()]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be a list of lists of `str` (list of tokenized texts).
        Return value is a list of `str` containing different amounts of the
        words "Positiv_Positiv", "Negativ_Negativ" based on the sentiments
        given to the input words by the Hardvard Inquirer lexicon.
        """
        newlist = [
            list(self._get_sentiment(str)) for str in X
        ]
        return newlist

    def _get_sentiment(self, str):
        corpus = self._get_corpus()
        for i in range(len(str)-1):
            prev = str[i].lower()
            word = str[i + 1].lower()
            sent = corpus.get(word.lower(), "")
            if sent != "":
                yield sent

    def _get_corpus(self):
        """
        Private method used to cache a dictionary with the Harvard Inquirer
        corpus.
        """
        if not self._corpus:
            corpus = defaultdict(str)
            with open('data/inquirerbasicttabsclean') as csvfile:
                it = csv.reader(csvfile, delimiter="\t")
                next(it)  # Drop header row
                for row in it:
                    entry = InquirerLexEntry(*row)
                    xs = []
                    for i in self._use_fields:
                        name, x = FIELDS[i], entry[i]
                        if x:
                            xs.append("{}_{}".format(name[0], x[0]))
                    name = entry.Entry.lower()
                    if "#" in name:
                        name = name[:name.index("#")]
                    corpus[name] = " ".join(xs)
                self._corpus.append(dict(corpus))
        return self._corpus[0]
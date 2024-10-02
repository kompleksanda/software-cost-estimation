from scipy.io import arff
from io import StringIO
import pandas as pd
from deap import gp
import numpy as np
import operator
import random
import math

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

class ArffParser:
    def __init__(self, path):
        self.raw_data = StringIO(open(path).read())
        self.parse()

    def parse(self):
        data, meta = arff.loadarff(self.raw_data)
        self.data_frame = pd.DataFrame(data)

class AlbrechtParser(ArffParser):
    def prepare(self):

        # extract effort, log the effort and replace it
        effort_col = self.data_frame['Effort']
        self.data_frame = self.data_frame.drop(labels=['Effort'], axis=1)
        self.data_frame.insert(len(self.data_frame.columns), 'Effort', effort_col)
        self.data_frame.Effort = np.log1p(self.data_frame.Effort)

        primitive_set = gp.PrimitiveSet("main", 7)
        primitive_set.addPrimitive(max, 2)
        primitive_set.addPrimitive(operator.add, 2)
        primitive_set.addPrimitive(operator.mul, 2)
        primitive_set.addPrimitive(operator.sub, 2)
        primitive_set.addPrimitive(min, 2)
        primitive_set.addPrimitive(math.cos, 1)
        primitive_set.addPrimitive(math.sin, 1)
        primitive_set.addEphemeralConstant("rand101", lambda: random.randint(0, 100))
        primitive_set.renameArguments(ARG0="Input")
        primitive_set.renameArguments(ARG1="Output")
        primitive_set.renameArguments(ARG2="Inquiry")
        primitive_set.renameArguments(ARG3="File")
        primitive_set.renameArguments(ARG4="FPAdj")
        primitive_set.renameArguments(ARG5="RawFPcounts")
        primitive_set.renameArguments(ARG6="AdjFP")
        
        return self.data_frame, primitive_set

class ChinaParser(ArffParser):
    def prepare(self):
        self.data_frame = self.data_frame.drop(columns="ID")
        self.data_frame = self.data_frame.drop(columns='Dev.Type')
        self.data_frame = self.data_frame.drop(columns='N_effort')

        # extract effort, log the effort and replace it
        effort_col = self.data_frame['Effort']
        self.data_frame = self.data_frame.drop(labels=['Effort'], axis=1)
        self.data_frame.insert(len(self.data_frame.columns), 'Effort', effort_col)
        self.data_frame.Effort = np.log1p(self.data_frame.Effort)

        primitive_set = gp.PrimitiveSet("main", 15)
        primitive_set.addPrimitive(operator.add, 2)
        primitive_set.addPrimitive(operator.sub, 2)
        primitive_set.addPrimitive(operator.mul, 2)
        primitive_set.addPrimitive(protectedDiv, 2)
        primitive_set.addPrimitive(operator.neg, 1)
        primitive_set.addEphemeralConstant("rand101", lambda: random.randint(0, 100))
        primitive_set.renameArguments(ARG0="AFP")
        primitive_set.renameArguments(ARG1="Input")
        primitive_set.renameArguments(ARG2="Output")
        primitive_set.renameArguments(ARG3="Enquiry")
        primitive_set.renameArguments(ARG4="File")
        primitive_set.renameArguments(ARG5="Interface")
        primitive_set.renameArguments(ARG6="Added")
        primitive_set.renameArguments(ARG7="Changed")
        primitive_set.renameArguments(ARG8="Deleted")
        primitive_set.renameArguments(ARG9="PDR_AFP")
        primitive_set.renameArguments(ARG10="PDR_UFP")
        primitive_set.renameArguments(ARG11="NPDR_AFP")
        primitive_set.renameArguments(ARG12="NPDU_UFP")
        primitive_set.renameArguments(ARG13="Resource")
        primitive_set.renameArguments(ARG14="Duration")
        
        return self.data_frame, primitive_set
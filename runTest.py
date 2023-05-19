from utils import *
from names import *
from preprocessing import *
from input import *
from modelling import *




def run():
    print("START RUN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prep_run()
    model = model_run(data, device)
    print("START RUN")


run()


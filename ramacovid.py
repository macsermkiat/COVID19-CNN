# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:32:32 2020

@author: ratch
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
#from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri
from sklearn.svm import SVR
from tqdm import tqdm

#pandas2ri.activate()

class ramacovid:
    
    def __init__(self, path):
        self.path = path
    
    def get_incidence(self, url):
        """
        This function retrieves daily incidence categorized by provinces

        Parameters
        ----------
        url : String
            API to Department of Disease Control.

        Returns
        -------
        incd : DataFRame
            Daily incidence categorized by provinces.

        """
        try:
            response = requests.get(url, timeout=10)
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
        print("requests code : {}".format(response.status_code))
        
        
        df = pd.read_json(json.dumps(response.json()["Data"]),orient="records",convert_dates=["ConfirmDate"])
        df = df.join(pd.get_dummies(df["ProvinceEn"]))
        incd= df.groupby("ConfirmDate").sum().iloc[:, 3:].reset_index()
        incd = incd.groupby("ConfirmDate").sum().iloc[:, 1:].reset_index()
        
        return incd
    '''
    def epiestim(self, df, mean=3.38, sd=1.4):
        """
        This function estimates Rt from EpiEstim given incidence

        Parameters
        ----------
        df : DataFrame
            Incidence (1st column is date, 2nd column is incidence)
        mean : Float, optional
            Mean of reproduction number (COVID-19). The default is 3.38.
        sd : Float, optional
            Standard deviation of reproduction number (COVID-19). The default is 1.4.

        Returns
        -------
        rhat : DataFrame
            Estimated reproduction number.

        """
        package_name = "EpiEstim"

        try:
            eps = importr(package_name)
        except:
            ro.r(f'install.packages("{package_name}")')
            eps = importr(package_name)

        rdf = pandas2ri.py2ri(df)
        results = eps.estimate_R(rdf[1], method="parametric_si", config=eps.make_config(mean_si=mean, std_si=sd))
        results = dict(results.items())
        rhat = pandas2ri.ri2py(results["R"])
        # Later will use python to call EpiEstim directly from R
        return rhat
    
    def get_dailyR(self, df):
        df["ConfirmDate"] = df["ConfirmDate"].apply(lambda x: x.strftime("%Y-%m-%d"))
        provinces = df.columns.tolist()[1::]
        rdaily = pd.DataFrame()
        for province in tqdm(provinces, total=len(provinces)):
            temp = df.loc[:, ["ConfirmDate", province]]
            temp[province] = temp[province].astype("int64")
            rhat = cov.epiestim(temp)
            rdaily[province] = rhat["Mean(R)"].tolist()
        rdaily["start"] = rhat["t_start"].tolist()
        rdaily["end"] = rhat["t_end"].tolist()
        return rdaily
    '''
    # def read_datasources(self, countries, window_size=7):
    #     incd = pd.read_csv(r"..\data\incidence.csv", parse_dates=["date"])
    #     rdaily = pd.read_csv(r"..\data\rdaily_gamma.csv")
        
    #     date = incd["date"].shift(-window_size).dropna().tolist()
    #     start_date = incd["date"][0:-window_size].tolist()
    #     end_date = incd["date"][window_size - 1:-1].tolist()
        
    #     robs = []
    #     for i in incd.index:
    #         if i >= 7:
    #             robs.append(incd["incidence"][i - window_size: i + 1].sum() / window_size)
        
    #     data = pd.DataFrame({"start": start_date, "end": end_date, "today": date, 
    #                          "Rhat": rdaily["Mean(R)"].tolist(), "Robs": robs})
        
    #     data["error"] = data["Rhat"] - data["Robs"]
        
    #     mdata = pd.read_csv(r"..\data\applemobilitytrends-2020-04-21.csv")
        
    #     mdata = mdata[mdata["region"].isin(countries)].reset_index(drop=True)
        
    #     days = mdata.columns.tolist()
    #     days.remove("geo_type")
    #     days.remove("region")
    #     days.remove("transportation_type")
    #     mobility = pd.DataFrame()
    #     for i in mdata.index:
    #         col_name = mdata.loc[i, "region"] + "_" + mdata.loc[i, "transportation_type"]
    #         density = mdata.loc[i, days]
    #         density = [d/100 for d in density]
    #         mobility[col_name] = density
        
    #     mobility["today"] = pd.to_datetime(days)
        
    #     data = pd.merge(data, mobility, on="today", how="left")
        
    #     densities = []
    #     countries = []
    #     transportations = []
    #     dates = []
    #     for i in mdata.index:
    #         density = mdata.loc[i, days].tolist()
    #         country = [mdata.loc[i, "region"]] * len(density)
    #         trans = [mdata.loc[i, "transportation_type"]] * len(density)
    #         densities = densities + density
    #         countries = countries + country
    #         transportations = transportations + trans
    #         dates = dates + days
        
    #     trans_data = pd.DataFrame({"today": dates, "country": countries, "trans_mode": transportations, "relative_change": densities})
    #     trans_data["today"] = pd.to_datetime(trans_data["today"])
    #     return data, trans_data
    
    def get_dataset(self, df, col):
        predictors = [col, "Rhat"]
        data = df.loc[:, predictors]
        data = data.reset_index()
        data = data.rename(columns={"index": "days"})
        
        predictors.remove("Rhat")
        
        X = data.loc[:, predictors].values
        y = data["Rhat"].values.reshape(-1, 1)
        
        return X, y
    
    def sim_data(self):
        
        location = {"Bangkok": {"Dusit": ("Suan Chit Lada", "Wachiraphayaban"),
                                "Ladprao": ("Ladprao", "Chorakhe Bua")},
                    "Nonthaburi": {"Pakkred": ("Ko Kret", "Tha It"),
                                   "Bangkruei": ("Sala Klang", "Maha Sawat")},
                    "Korat": {"Phimai": ("Dong Yai", "Tha Luang"),
                              "Khong": ("Nong Bua", "Ta Chan")}}
        
        df = pd.DataFrame()
        for province in location.keys():
            for district in location[province].keys():
                for subdistrict in location[province][district]:
                    temp = pd.DataFrame({"date": pd.date_range("2020-01-12", "2020-04-27")})
                    temp = temp.reset_index()
                    temp = temp.rename(columns={"index": "day"})
                    temp["province"] = province
                    temp["district"] = district
                    temp["subdistrict"] = subdistrict
                    temp["rel_change"] = np.random.randint(-10000, 10000, size=len(temp.index))
                    temp["mobility"] = np.random.randint(10, 10000, size=len(temp.index))
                    temp["driving"] = np.random.randint(1, 10, size=len(temp.index))
                    temp["walking"] = np.random.randint(1, 10, size=len(temp.index))
                    temp["transit"] = np.random.randint(1, 10, size=len(temp.index))
                    temp["sum"] = temp["driving"] + temp["walking"] + temp["transit"]
                    temp["driving"] = temp["driving"] / temp["sum"]
                    temp["walking"] = temp["walking"] / temp["sum"]
                    temp["transit"] = temp["transit"] / temp["sum"]
                    temp = temp.drop("sum", axis=1)
                    temp["age_mean"] = np.random.uniform(20, 70, len(temp.index))
                    temp["age_sd"] = np.random.uniform(1, 10, len(temp.index))
                    temp["male"] = np.random.uniform(0.1, 0.9, len(temp.index))
                    temp["female"] = 1 - temp["male"]
                    temp["prepaid"] = np.random.uniform(0.1, 0.9, len(temp.index))
                    temp["postpaid"] = 1 - temp["prepaid"]
                    
                    df = pd.concat([df, temp], axis=0, ignore_index=True, sort=False)
        return df

if __name__ == "__main__":
    cov = ramacovid(r"G:\Shared drives\RamaCovid")
    
    incd = cov.get_incidence(url="https://covid19.th-stat.com/api/open/cases")
    # rdaily = cov.get_dailyR(df=incd)
    
    samples = cov.sim_data()
    
    
    
    
    
    
    
    
    
    # df, tdf = cov.read_datasources(countries=("Denmark", "Thailand", "Singapore", "Japan"))
    
    
    # mobile = "Thailand_driving"
    # ml = "SVR-RBF"
    # X, y = cov.get_dataset(df, mobile)
    
    # clf = SVR(C=1.0, epsilon=0.2)
    # clf.fit(X, y)
    
    # yhat = clf.predict(X)
    
    # results = pd.DataFrame({mobile: df[mobile], "Rhat": df["Rhat"], "ml": yhat})
    # results = results.sort_values(by=mobile).reset_index(drop=True)
    # results["error"] = results["Rhat"] - results["ml"]
    # error = (results["Rhat"] - results["ml"]).sum()
    
    # plt.plot(results[mobile], results["ml"], color="blue", label="R - " + ml, linewidth=1)
    # # plt.fill_between(results[mobile], results["ml"] - error, results["ml"] + error, alpha=1, facecolor="pink")
    # plt.scatter(results[mobile], results["Rhat"])
    # plt.xlabel("Mobile - " + mobile)
    # plt.ylabel("Rt (EpiEstim)")
    # plt.legend()
    # plt.show()
    # tdf.to_csv("../data/trans_data.csv", index=False)
    
    

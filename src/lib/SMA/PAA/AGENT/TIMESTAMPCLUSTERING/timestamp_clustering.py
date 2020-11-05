""" Timestamp clustering API for Port Activity App """

import os
import json

from datetime import timedelta
from dateutil import parser

import requests

from flask import Flask
from flask_restful import Resource, Api

import pandas as pd
import numpy as np
import scipy.signal
from sklearn.neighbors import KernelDensity
#import matplotlib.pyplot as plt


class TimestampFethcer:
    """ Fetch timestamps from Port Activity App API """
    # pylint: disable=R0903

    def __init__(self, imo, url, api_key):
        self.imo = imo
        self.url = url
        self.api_key = api_key

    def fetch(self):
        """ Fetch timestamps from Port Activity App API """
        timestamps = []
        headers = {'Authorization': 'ApiKey ' + self.api_key}
        offset = 0
        limit = 100

        while True:
            payload = {'imo': self.imo, 'offset': offset, 'limit': limit}
            request = requests.get(self.url, params=payload, headers=headers)
            response = json.loads(request.text)

            if not 'data' in response:
                return []

            datas = response['data']
            for data in datas:
                timestamps.append(data)

            offset = offset + limit
            if offset >= response['pagination']['total']:
                break

        return timestamps


class KernelDensityEstimator:
    """ Kernel density estimation to cluster timestamps """
    # pylint: disable=R0903

    def __init__(self):
        self.bandwidth_factor = 50

    def calculate_cluster_ids(self, timestamps):
        """ Calculate cluster IDs """
        # pylint: disable=C0103,R0914,W0631
        df = pd.read_json(json.dumps(timestamps))
        df["t"] = df.time.apply(parser.parse)
        df["t"] = (df["t"] - df["t"].min()) / pd.to_timedelta(1, unit='D')
        bandwith = (df.t.max() - df.t.min()) / self.bandwidth_factor
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwith)
        kde.fit(df.t.values.reshape(-1, 1))
        timestamp_axis = np.arange(df.t.min(), df.t.max(), bandwith / 100)
        density_estimate = kde.score_samples(timestamp_axis.reshape(-1, 1))
        peaks_data = scipy.signal.find_peaks(-density_estimate)
        minima_indices = peaks_data[0]
        t = df.t
        x = np.arange(len(t))
        cluster_ids = list()
        for minimum_index in minima_indices:
            b = t < timestamp_axis[minimum_index]
            xb = x[b]
            cluster_ids.extend([minimum_index for i in range(len(xb))])
            #tb = t[b]
            x = x[~b]
            t = t[~b]
            #plt.plot(xb, tb, "-o")
        cluster_ids.extend([minimum_index + 1 for i in range(len(x))])
        #plt.plot(x, t, "-o")
        # plt.show()

        return cluster_ids


class TimestampPostProcessor:
    """ Post processing of timestamps """
    # pylint: disable=R0201,R0903

    def process(self, timestamps):
        """ Post processing of timestamps """
        idx = 0
        cluster_id = timestamps[0]['cluster_id']
        temp_port_call_id = 0
        actual_departure = False
        # pylint: disable=R1702
        for timestamp in timestamps:
            # pylint: disable=E1101
            # Store actual departure time
            if timestamp['time_type'] == 'Actual' and 'Departure_Vessel_' in timestamp['state']:
                actual_departure = True
                actual_departure_time = parser.parse(timestamp['time'])

            # Check for cluster break
            if cluster_id != timestamp['cluster_id']:
                app.logger.info('--- Cluster break ---')
                cluster_id = timestamp['cluster_id']
                # If we have actual departure then close port call
                if actual_departure:
                    app.logger.info('--- Actual departure resolved ---')
                    temp_port_call_id += 1
                    actual_departure = False
                # We do not have actual departure, more resolving needed
                elif not actual_departure:
                    app.logger.info('--- Actual departure not resolved ---')
                    future_departure = False
                    future_arrival = False
                    # Scan next cluster if it has departure without arrival
                    future_idx = idx
                    for future_timestamp in timestamps[idx:]:
                        future_idx += 1
                        if 'Arrival_Vessel_' in future_timestamp['state']:
                            future_arrival = True
                        if 'Departure_Vessel_' in future_timestamp['state']:
                            future_departure = True
                        # End of future cluster
                        if cluster_id != future_timestamp['cluster_id'] \
                                or future_idx == len(timestamps):
                            app.logger.info('--- Future cluster break ---')
                            # We have future cluster with arrival or without departure
                            # Close current port call since next cluster is new port call
                            if future_arrival or not future_departure:
                                app.logger.info(
                                    '--- No departure in future ---')
                                temp_port_call_id += 1
                                actual_departure = False
                            break
            # There is no cluster break but we have actual departure and timestamp is arrival
            elif actual_departure and 'Arrival_Vessel_' in timestamp['state']:
                #app.logger.info('--- Pre arrival after actual departure ---')
                current_time = parser.parse(timestamp['time'])
                # If new arrival is farther than 1 day from actual departure
                # then close this port call
                if (current_time - actual_departure_time) > timedelta(days=1):
                    app.logger.info('--- Arrival after actual departure ---')
                    temp_port_call_id += 1
                    actual_departure = False

            timestamp['temp_port_call_id'] = temp_port_call_id
            idx += 1

            app.logger.info(str(timestamp['cluster_id']) +
                            ": " + str(timestamp['temp_port_call_id']) +
                            ": " + timestamp['time'] +
                            " " + timestamp['time_type'] +
                            " " + timestamp['state'] +
                            " (" + timestamp['created_at'] + ")")


class TimestampClusterer:
    """ Clusters timestamps for given IMO """

    def __init__(self, imo):
        self.imo = imo
        self.timestamps = []
        self.cluster_ids = []

    def cluster(self):
        """ Clusters timestamps for given IMO """
        url = os.getenv('API_TIMESTAMP_EXPORT_URL')
        api_key = os.getenv('API_KEY')

        if url is None or api_key is None:
            return {'result': 'ERROR'}

        self.fetch_timestamps(url, api_key)

        if len(self.timestamps) == 0:
            return {'result': 'ERROR'}

        self.add_dummy_timestamps()
        self.calculate_cluster_ids()
        self.attach_cluster_ids()
        self.remove_dummy_timestamps()
        self.post_process()

        return self.output()

    def fetch_timestamps(self, url, api_key):
        """ Fetch timestamps """
        fethcer = TimestampFethcer(self.imo, url, api_key)
        self.timestamps = fethcer.fetch()

    def add_dummy_timestamps(self):
        """ Add dummy timestamps to beginning and end """
        start_time = parser.parse(self.timestamps[0]['time'])
        end_time = parser.parse(self.timestamps[-1]['time'])
        dummy_start_time = (start_time - timedelta(days=30)
                            ).strftime("%m-%d-%Y %H:%M:%S+00")
        dummy_end_time = (end_time + timedelta(days=30)
                          ).strftime("%m-%d-%Y %H:%M:%S+00")
        self.timestamps.insert(
            0,
            {'time_type': 'Dummy',
             'state': 'Dummy',
             'created_at': 'Dummy',
             'time': dummy_start_time}
        )
        self.timestamps.append(
            {'time_type': 'Dummy',
             'state': 'Dummy',
             'created_at': 'Dummy',
             'time': dummy_end_time}
        )

    def calculate_cluster_ids(self):
        """ Calculate cluster IDs """
        estimator = KernelDensityEstimator()
        self.cluster_ids = estimator.calculate_cluster_ids(self.timestamps)

    def attach_cluster_ids(self):
        """ Attach cluster IDs to timestamps """
        idx = 0
        for cluster_id in self.cluster_ids:
            self.timestamps[idx]['cluster_id'] = cluster_id
            idx += 1

    def remove_dummy_timestamps(self):
        """ Remove dummy timestamps from beginning and end """
        self.timestamps = self.timestamps[1:len(self.timestamps)-1]

    def post_process(self):
        """ Post process timestamp clustering """
        post_processor = TimestampPostProcessor()
        post_processor.process(self.timestamps)

    def output(self):
        """ Output clustered timestamps """
        output = dict()
        for timestamp in self.timestamps:
            output[timestamp['id']] = timestamp['temp_port_call_id']

        return output


class HealthCheck(Resource):
    """ Alive check """
    #pylint: disable=R0201

    def get(self):
        """ Alive check """
        return {'result': 'OK'}


class ClusterTimestamps(Resource):
    """ GET for timestamp clustering, IMO as input argument """
    #pylint: disable=R0201

    def get(self, imo):
        """ GET for timestamp clustering, IMO as input argument """
        clusterer = TimestampClusterer(imo)
        result = clusterer.cluster()
        return result


app = Flask(__name__)
api = Api(app)

api.add_resource(HealthCheck, '/')
api.add_resource(ClusterTimestamps, '/cluster-timestamps/<int:imo>')

if __name__ == '__main__':
    if os.getenv('API_TIMESTAMP_EXPORT_URL') is None:
        raise Exception('API_TIMESTAMP_EXPORT_URL not defined')
    if os.getenv('API_KEY') is None:
        raise Exception('API_KEY not defined')

    app.run(host='0.0.0.0', debug=True)

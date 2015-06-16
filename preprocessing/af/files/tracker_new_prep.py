import os
import re
import numpy as np
import pandas as pd
import tables as tb
from scipy.sparse import rand, coo_matrix

import timeit

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

log_folder = r'logs/af/'
processed_folder = r'logs/processed/'


def get_file_paths(folder=processed_folder):
    """
    Returns List holding paths of all .txt Files in passed Folder.
    """
    text_files = []
    for text_file in os.listdir(folder):
        if text_file.endswith('.txt'):
            logfile_path = folder + text_file
            text_files.append(logfile_path)
    return text_files


def remove_milliseconds():
    """
    Adapts the TimeStamps for pandas default Time Parser.
    """
    file_paths = get_file_paths(log_folder)
    find_regex = r'(?<=[0-2][0-9]:[0-5][0-9]:[0-5][0-9]),[0-9]{3}'
    replacement_string = r''
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        output_path = processed_folder + os.path.split(logfile_path)[1] + '.txt'
        with open(output_path, 'w') as output_f:
            with open(logfile_path, 'r') as logfile:
                for line in logfile:
                    processed_line = re.sub(find_regex, replacement_string, line)
                    output_f.write(processed_line)


def load_file_into_df():
    """
    Loads Logfile; Assigns Column Names; strips whitespaces; Saves DataFrame
    """
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=None, parse_dates=['Date'], infer_datetime_format=True,
                         names=['Date',
                                'Method',
                                'Status',
                                'Servername',
                                'Request-URI',
                                'Request-Query',
                                'Content-Type',
                                'Session-ID',
                                'Remote-IP',
                                'User-Name-Hash',
                                'Referer',
                                'User-Agent'])
        # not all are stripped
        df['Method'] = df['Method'].str.strip(' ')
        df['Servername'] = df['Servername'].str.strip(' ')
        df['Request-URI'] = df['Request-URI'].str.strip(' ')
        df['Request-Query'] = df['Request-Query'].str.strip(' ')
        df['Content-Type'] = df['Content-Type'].str.strip(' ')
        df['Session-ID'] = df['Session-ID'].str.strip(' ')
        df['Remote-IP'] = df['Remote-IP'].str.strip(' ')
        df['User-Name-Hash'] = df['User-Name-Hash'].str.strip(' ')
        df['Referer'] = df['Referer'].str.strip(' ')
        df['User-Agent'] = df['User-Agent'].str.strip(' ')

        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
        del df


def find_useragent_bots():
    """
    Scans User-Agent for: bot, slurp, crawl, spider.
    Stored found User-Agents in bots.txt (appends to set if already exists).
    """
    bots = set(line.strip() for line in open('data/bots.txt'))
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0)
        df_mask = df['User-Agent'].str.lower().str.contains('bot', na=False)
        bots |= set(df['User-Agent'][df_mask])
        df_mask = df['User-Agent'].str.lower().str.contains('slurp', na=False)
        bots |= set(df['User-Agent'][df_mask])
        df_mask = df['User-Agent'].str.lower().str.contains('crawl', na=False)
        bots |= set(df['User-Agent'][df_mask])
        df_mask = df['User-Agent'].str.lower().str.contains('spider', na=False)
        bots |= set(df['User-Agent'][df_mask])

    bots = sorted(bots)
    with open('data/bots.txt', 'w') as bot_file:
        for name in bots:
            bot_file.write(str(name) + '\n')


def remove_bot_entries():
    """
    Reads User-Agents from bots.txt by row and removes all matching Entries.
    """
    bots = set()
    with open('data/bots.txt', 'r') as bot_file:
        for line in bot_file.readlines():
            bots.add(line.rstrip())
    file_paths = get_file_paths(processed_folder)
    total_bot_requests = 0

    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)

        df_len = len(df.index)
        df_mask = df['User-Agent'].isin(bots)
        df = df[~df_mask]
        bot_requests = df_len - len(df.index)
        total_bot_requests += bot_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Total Bot Requests: " + str(total_bot_requests)


def count_xml_requests(remove_them=False):
    """
    Counts (and removes if desired) Requests with Content-Type: xml --> (only used by crawlers?)
    """
    xml_requests = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_mask = df['Content-Type'] == 'text/xml;charset=ISO-8859-1'
        df_len = len(df.index)
        df = df[~df_mask]
        xml_requests += df_len - len(df.index)
        if remove_them:
            df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
        del df
    if remove_them:
        print "Found amd removed" + str(xml_requests) + " Requests"
    else:
        print "Found " + str(xml_requests) + " Requests"


def filter_html_requests():
    """
    Remove all requests which are not in variable 'html_requests'
    """
    non_html_requests = 0
    html_requests = ['text/html',
                     'text/html;charset=ISO-8859-1',
                     'text/html;charset=UTF-8',
                     'text/plain;charset=utf-8',
                     'text/xml;charset=ISO-8859-1']
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_mask = df['Content-Type'].isin(html_requests)
        df_len = len(df.index)
        df = df[df_mask]
        removed_requests = df_len - len(df.index)
        non_html_requests += removed_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Removed " + str(non_html_requests) + " Requests"


def print_jsp_requests():
    """
    Prints Referer Requests which end on .jsp to console.
    """
    jsp_requests = set()
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df = df.dropna(subset=['Referer'])
        df_mask = df['Referer'].str.endswith('.jsp')
        series = df['Referer'][df_mask]
        jsp_requests |= set(series)
    print jsp_requests


def filter_referer():
    """
    Removes Referers which are not representative for user behaviour
    """
    bad_queries = ['/admin/Admin.jsp',
                   '/admin/gwurz/BilderWurz.jsp',
                   '/admin/gwurz/FindWB.jsp']

    """
    Others:
                   #'/Edit.jsp'
                   #'/BiographySearch.jsp',
                   #'/Bookmark.jsp',
                   #'/geocreator/mapcreator/bildercountry_map.jsp',
                   #'/geocreator/story/comparison/bildercountry_story.jsp',
                   #'/geocreator/story/comparison2/bildercountry_story.jsp',
                   #'/Login.jsp',
                   #'/work/bilder/geo/web/story/bildercountry_story.jsp'
    """
    total_bad_queries = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        # TODO: Move NaNs to separate DF and combine DFs after filtering
        df = df.dropna(subset=['Referer'])
        df_mask = df['Referer'].isin(bad_queries)
        """
        should I remove the Edits ? Ask Daniel
        store session number where Edit was called and remove complete session (Admin vs User behaviour?)
        """
        #df_mask = df['Request-URI'].map(lambda x: x.startswith(r'/Edit.jsp'))
        df_len = len(df.index)
        df = df[~df_mask]
        removed_requests = df_len - len(df.index)
        total_bad_queries += removed_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Removed " + str(total_bad_queries) + " Requests"


def filter_requests_uri():
    """
    Removes all kinds of Request-URIs, mostly Admin Stuff.
    TODO: Maybe rethink: GoogleMap, UserPreferences ?
    """
    bad_queries = ['/JSON-RPC',
                   '/admin/gwurz/LogViewer.jsp',
                   '/admin/gwurz/Stats.jsp',
                   '/admin/gwurz/ToggleLogger.jsp',
                   '/Delete.jsp',
                   '/Diff.jsp',
                   '/geolab/prop1.jsp',
                   '/GetRenameLog.jsp',
                   '/GoogleMap.jsp',
                   '/LoginWS.jsp',
                   '/PageModified.jsp',
                   '/Preview.jsp',
                   '/Rename.jsp',
                   '/rss.jsp',
                   '/templates/default/AJAXTemplate.jsp',
                   '/Upload.jsp',
                   '/UploadPopup.jsp',
                   '/UserPreferences.jsp']
    total_bad_queries = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_mask = df['Request-URI'].isin(bad_queries)
        df_len = len(df.index)
        df = df[~df_mask]
        removed_requests = df_len - len(df.index)
        total_bad_queries += removed_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Removed " + str(total_bad_queries) + " Requests"


def filter_status_codes():
    """
    Only keep Requests with Status Code 200.
    """
    total_bad_status = 0
    status_codes_accepted = [200]
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_mask = df['Status'].isin(status_codes_accepted)
        df_len = len(df.index)
        df = df[df_mask]
        removed_requests = df_len - len(df.index)
        total_bad_status += removed_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Total Bad Status Requests removed: " + str(total_bad_status)


def filter_servername():
    """
    Removes 'www.'-Prefix from the Servername field.
    """
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df['Servername'] = df['Servername'].str.lower()
        df['Servername'] = df['Servername'].str.lstrip('www.')
        df['Servername'] = df['Servername'].str.rstrip('.')
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)


def filter_admin_requests():
    """
    Removes all Entries where the Request-URI contains '/admin'.
    """
    admin_requests = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0)
        df_len = len(df.index)

        df_mask = df['Request-URI'].str.contains('/admin', na=False)
        df = df[~df_mask]

        request_count = df_len - len(df.index)
        admin_requests += request_count
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Total admin requests: " + str(admin_requests)


def filter_af_proxy_ip():
    """
    Removes all Request coming from AF Proxy
    """
    af_staff_requests = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0)
        df_len = len(df.index)
        df_mask = df['Remote-IP'] == '129.27.200.50'  # AF Proxy
        df = df[~df_mask]
        removed_requests = df_len - len(df.index)
        af_staff_requests += removed_requests
        #df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Total admin requests: " + str(af_staff_requests)


def add_time_delta():
    """
    Adds a Column which shows the Time passed between Requests.
    First requests in a session have delta=0, therefor 'average session length' must be calculated with session_length-1
    """
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)

        df.set_index(['Session-ID', 'Remote-IP'], inplace=True)
        df = df.sortlevel(level='Session-ID', sort_remaining=True)
        df['TimeDelta'] = pd.Timedelta(days=0, seconds=0)
        delta_list = []
        for session_id in df.index.levels[0]:
            session = df.loc[session_id]
            prev_timestamp = 0
            for time_stamp in session['Date']:
                if prev_timestamp == 0:
                    delta_list.append(pd.Timedelta(days=0, seconds=0))
                else:
                    delta_list.append(time_stamp - prev_timestamp)
                prev_timestamp = time_stamp

        df['TimeDelta'] = delta_list
        df = df[['Date',
                 'TimeDelta',
                 'Method',
                 'Status',
                 'Servername',
                 'Request-URI',
                 'Request-Query',
                 'Content-Type',
                 'User-Name-Hash',
                 'Referer',
                 'User-Agent']]
        df.reset_index(inplace=True)
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)


def remove_non_af_entries():
    """
    TODO: Do I still need this? If yes, I also need to add those URLs to the prefix removal process)
    """
    total_removes = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df['Referer'] = df['Referer'].astype('string')
        df_mask = df['Referer'].map(lambda x: x.startswith(r'http://austria-lexikon.at'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.austria-lexikon.at'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.austria-forum.at'))

        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.austria-forum.org'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://austria-forum.org'))

        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.austria-forum.iicm.tu-graz.ac.at'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://austria-forum.iicm.tu-graz.ac.at'))

        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.virt4.iicm.tu-graz.ac.at'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://virt4.iicm.tu-graz.ac.at'))

        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://geographyoftheworld.org'))
        df_mask = df_mask | df['Referer'].map(lambda x: x.startswith(r'http://www.geographyoftheworld.org'))
        # todo: check whether those are all (remove http:// ?)

        df_len = len(df.index)
        df = df[df_mask]
        removed_requests = df_len - len(df.index)
        total_removes += removed_requests
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)
    print "Removed: " + str(total_removes)


def count_pages():
    """
    Combines all DFs; Counts the Page Visits; Stores them to File.
    """
    page_count_series = pd.Series(name='Page')

    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=0):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        page_count_series = page_count_series.append(df['Request-URI'])
        del df

    page_count_series = page_count_series.value_counts()
    print "Total AF Page Calls: " + str(page_count_series.sum())
    print "Calls per Day: " + str(page_count_series.sum() / 37)
    print "Distinct Pages: " + str(len(page_count_series))

    page_counts_df = pd.DataFrame(data=dict(ID=xrange(0, len(page_count_series.index), 1),
                                            Page=page_count_series.index,
                                            Visits=page_count_series))
    page_counts_df.to_csv('logs/page_count_df.csv', header=True, index=False)


def remove_unneeded_columns():
    """
    After initial Calculations, those columns are not needed any more and take up space.
    Remove them so we can combine the DFs.
    """
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df = df.drop(['Method',
                      'Status',
                      'Servername',
                      'Request-Query',
                      'Content-Type',
                      'User-Name-Hash',
                      'User-Agent'], axis=1)
        df.to_csv(path_or_buf=logfile_path, sep='|', header=True, index=False)


def add_page_id():
    """
    If page is from AF, add the page id from page_count_df.
    Otherwise, set Page-ID NaN
    """
    page_count_df = pd.read_csv('logs/page_count_df.csv', header=0)
    page_count_df = page_count_df.drop(['Visits'], axis=1)
    page_count_df.set_index(['Page'], inplace=True)

    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)

        # Has been done before. Better safe than sorry.
        df['Referer'] = df['Referer'].str.lstrip()
        # remove 'http://austria-forum.org' prefix from referer
        # some nan values in there (floats), no need to remove strings there
        df['Referer'] = [remove_url_prefix(x) for x in df['Referer']]

        # set id's where possible
        df['Request-ID'] = [map_string_to_id(x, page_count_df) for x in df['Request-URI']]
        df['Referer-ID'] = [map_string_to_id(x, page_count_df) for x in df['Referer']]

        # rearrange
        df = df[['Session-ID', 'Remote-IP', 'Date', 'TimeDelta', 'Referer-ID', 'Referer', 'Request-ID', 'Request-URI']]
        df.to_csv(logfile_path, sep='|', header=True, index=False)


def remove_url_prefix(line):
    """
    Helper Function for add_page_id()
    AF Prefix occurs only in Referer, some of them are Empty --> check for NaN.
    If Prefix occurs, return substring.
    TODO: What about the other AF Domains (geographyoftheworld.org, ...)
    """
    if isinstance(line, float) and np.isnan(line):
        return
    if line.startswith('http://austria-forum.org'):
        return line[len('http://austria-forum.org'):]
    else:
        return line


def map_string_to_id(item, page_count_df):
    """
    Helper Function for add_page_id()
    If items exists in our page index, map its ID and return it.
    Otherwise it's an external/empty referer.
    """
    if item is None:
        return np.nan
    try:
        item_id = page_count_df.loc[item, 'ID']
    except KeyError:
        item_id = np.nan
    return item_id


def combine_dataframe():
    """
    Combines all single files to one big DataFrame.
    """
    file_paths = get_file_paths(processed_folder)
    one_big_dataframe = pd.DataFrame()
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        one_big_dataframe = one_big_dataframe.append(df)
        del df
    one_big_dataframe.to_csv('logs/one_big_df.csv', sep='|', header=True, index=False)


def add_session_number_to_combined_df():
    """
    Combines daily DataFrames.
    Calculates Session-Nrs based on a Reference Time Delta
    """
    print "Reading big DF to File... ",
    big_df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
    print "done"
    big_df['TimeDelta'] = pd.to_timedelta(arg=big_df['TimeDelta'])

    big_df.set_index(['Session-ID'], inplace=True)
    # need to make entries distinct for selecting data
    big_df['idx'] = [x for x in xrange(0, len(big_df.index), 1)]
    big_df.set_index(keys=['idx', 'TimeDelta'], append=True, inplace=True)
    big_df = big_df.sortlevel(level='Session-ID', sort_remaining=True)

    big_df['Session-Nr'] = np.nan

    #print big_df.head(40)
    #print len(big_df.index.levels[0].unique())

    session_number = 0
    reference_delta = pd.Timedelta(days=0, minutes=30, seconds=0)
    session_number_array = []
    print "Extracting Sessions with Time Delta set to: " + str(reference_delta)
    start_time = timeit.default_timer()
    for session_id in big_df.index.levels[0]:
        session = big_df.loc[session_id]
        session_number += 1
        if len(session.index) == 1:
            #big_df.loc[session_id, 'Session-Nr'] = session_number
            session_number_array.append(session_number)  # this is much, much, much faster
        else:
            for index, time_delta in session.index:
                # first element in each session
                if time_delta > reference_delta:
                    session_number += 1
                #big_df.loc[(session_id, index), 'Session-Nr'] = session_number
                session_number_array.append(session_number)
        del session
        if session_number % 1000 == 0:
            print "Sessions done: " + str(session_number)
    print timeit.default_timer() - start_time

    big_df['Session-Nr'] = session_number_array

    # Rearrange Columns
    big_df = big_df.reset_index()
    big_df = big_df[['Session-ID',
                     'Remote-IP',
                     'Session-Nr',
                     'Date',
                     'TimeDelta',
                     'Referer-ID',
                     'Referer',
                     'Request-ID',
                     'Request-URI']]
    big_df['Session-Nr'] = big_df['Session-Nr'].astype('int64')
    print "Writing to File"
    big_df.to_csv('logs/big_session_df.csv', sep='|', header=True, index=False)

    number_of_requests = len(big_df.index)
    number_of_sessions = len(big_df['Session-Nr'].unique())
    average_session_length = float(number_of_requests) / float(number_of_sessions)
    print "Reference Delta: " + str(reference_delta)
    print "Number of Sessions: " + str(number_of_sessions)
    print "Average Session Length: " + str(average_session_length)


def count_all_clicks():
    """
    Counts clicks (=Requests/Transitions) from all Files.
    """
    clicks = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        clicks += len(df.index)
        del df
    print "# Clicks: " + str(clicks)


def count_session_ids():
    """
    Counts the number of unique Session-IDs across all files.
    """
    unique_session_ids = set()
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        unique_session_ids |= set(df['Session-ID'].unique())
        del df
    print "There are " + str(len(unique_session_ids)) + " unique Session-ID's"


def count_remote_ips():
    """
    Counts and prints the number of unique Remote-IPs
    Useful to compare it to the number of Session-IDs created by the Logger.
    """
    unique_remote_ips = set()
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        unique_remote_ips |= set(df['Session-ID'].unique())
        del df
    print "There are " + str(len(unique_remote_ips)) + " unique Remote-IP's"


def count_clicks_from_outside():
    """
    Counts the NaN in the Referer-ID.
    The Referer-ID is NaN when the Referer is not within mapped AF Pages.
    (e.g. Google Referer --> NaN ID)
    """
    all_requests = 0
    external_requests = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_len = len(df.index)
        all_requests += df_len
        # remove nan
        df = df[~np.isnan(df['Referer-ID'])]
        # count how many removed
        external_requests += df_len - len(df.index)
        del df
    print "There are " + str(all_requests) + " in total and " + str(external_requests) + " of those are external."


def count_nan_referer():
    """
    Counts the NaN in the Referer (String).
    This excludes Referrers like Google, only empty Referrers are counted (e.g. Bookmarks, Crawlers)
    """
    nan_referer = 0
    file_paths = get_file_paths(processed_folder)
    for count, logfile_path in enumerate(file_paths, start=1):
        print "File #" + str(count) + " -> " + os.path.split(logfile_path)[1]
        df = pd.read_csv(logfile_path, sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
        df_mask = [check_string_nan(x) for x in df['Referer']]

        # df_mask is somehow a list
        df_mask = pd.Series(df_mask)
        df = df[df_mask]
        nan_referer += len(df.index)
        del df
    print "There are " + str(nan_referer) + " Requests with an empty Referer"


def check_string_nan(line):
    """
    Helper Function for count_nan_referer().
    Need to do it this way; can't call .isnan on String (which most Referers are).
    """
    if isinstance(line, float) and np.isnan(line):
        return True
    else:
        return False


def split_single_click_session_from_big_df():
    """
    Splits the single Click Sessions from DF.
    Creates a new DataFrame which holds those Sessions.
    """
    big_df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0)
    session_lenghts = big_df['Session-Nr'].value_counts()
    session_lenghts = session_lenghts[session_lenghts.values == 1]
    df_mask = big_df['Session-Nr'].isin(session_lenghts.index)
    one_click_df = big_df[df_mask]
    big_df = big_df[~df_mask]
    big_df.to_csv('logs/big_session_df.csv', sep='|', header=True, index=False)
    one_click_df.to_csv('logs/one_click_sessions_df.csv', sep='|', header=True, index=False)


def split_non_af_entries():
    """
    Removes all entries where Referer-ID was not set (not Mapped).
    Creates Separate DataFrame for Removed Entries.
    """
    big_df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0)
    df_mask = np.isnan(big_df['Referer-ID'])
    non_af_df = big_df[df_mask]
    big_df = big_df[~df_mask]
    big_df.to_csv('logs/big_session_df.csv', sep='|', header=True, index=False)
    non_af_df.to_csv('logs/non_af_df.csv', sep='|', header=True, index=False)


def filter_long_sessions():
    """
    Removes all Sessions which fit the criteria of the following two variables:
    session_length_threshold : int
        All Sessions with # clicks greater than this threshold are considered.
    seconds_on_page : int
        The average time spent per page be is <= seconds_on_page.
    """
    session_length_threshold = 10
    seconds_on_page = 3

    big_df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0)
    session_lengths = big_df['Session-Nr'].value_counts()

    sessions_above_threshold = session_lengths[session_lengths.values > session_length_threshold]
    valid_count = 0
    invalid_session_nr = set()
    for count, session_nr in enumerate(sessions_above_threshold.index, start=1):
        session = big_df[big_df['Session-Nr'] == session_nr]
        overall_time = pd.Timedelta(seconds=0)
        for time_delta in session['TimeDelta']:
            overall_time = overall_time + time_delta
        # average time does not include first item (therefore -1)
        average_delta = overall_time / (len(session.index)-1)
        #print "Session Length:" + str(len(session.index)) + ", Average Time on Page: " + str(average_delta)
        if average_delta > pd.Timedelta(seconds=seconds_on_page):
            valid_count += 1
        else:
            invalid_session_nr.add(session_nr)
    print "# Sessions with length > " + str(session_length_threshold) + ": " + str(len(sessions_above_threshold.index))
    print "# Sessions with Avg. Time on Page > " + str(seconds_on_page) + ": " + str(valid_count)

    big_df = big_df[~big_df['Session-Nr'].isin(invalid_session_nr)]
    big_df.to_csv('logs/big_session_df.csv', sep='|', header=True, index=False)


def remove_trailing_slash():
    """
    /af/AEIOU/ --> /af/AEIOU
    To prevent different Mappings.
    """
    big_df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0)
    big_df['Referer'] = [string_ends_with_slash(x) for x in big_df['Referer']]
    big_df['Request-URI'] = [string_ends_with_slash(x) for x in big_df['Request-URI']]
    big_df.to_csv('logs/big_session_df.csv', sep='|', header=True, index=False)


def string_ends_with_slash(string):
    """
    Helper string-manipulation function for remove_trailing_slash()
    """
    # don't remove slash from main page
    if len(string) == 1:
        return string
    elif string.endswith(r'/'):
        # string without last character
        return string[:(len(string)-1)]
    else:
        return string


def count_pages_in_big_df():
    """
    Counts the Page Visits; Stores them to File.
    """
    big_df = pd.read_csv('logs/big_session_df.csv',
                         sep='|',
                         header=0,
                         parse_dates=['Date'],
                         infer_datetime_format=True)
    # we need those for the IDs since they can occur in referer
    external_clicks = pd.read_csv('logs/non_af_df.csv',
                                  sep='|',
                                  header=0,
                                  parse_dates=['Date'],
                                  infer_datetime_format=True)
    big_df = big_df.append(external_clicks)
    page_count_series = big_df['Request-URI']
    del big_df

    page_count_series = page_count_series.value_counts()
    print "Total AF Page Calls: " + str(page_count_series.sum())
    print "Calls per Day: " + str(page_count_series.sum() / 37)
    print "Distinct Pages: " + str(len(page_count_series))

    page_counts_df = pd.DataFrame(data=dict(ID=xrange(0, len(page_count_series.index), 1),
                                            Page=page_count_series.index,
                                            Visits=page_count_series))
    page_counts_df.to_csv('logs/page_count_big_df.csv', header=True, index=False)


def create_mapping_for_flo():
    # recounts the pages in big_df, uses \t separator
    page_count_df = pd.read_csv('logs/page_count_big_df.csv', header=0)
    page_count_df = page_count_df.drop(['Visits'], axis=1)
    page_count_df.to_csv('logs/id_name_mapping.csv', sep='\t', header=True, index=False)


def remap_big_df_page_ids():
    """
    Mapping of Page-ID -> URL has changed after removing Single Click Sessions. (Smaller Index)
    """
    page_count_df = pd.read_csv('logs/id_name_mapping.csv', sep='\t', header=0)
    page_count_df.set_index(['Page'], inplace=True)

    df = pd.read_csv('logs/big_session_df.csv', sep='|', header=0, parse_dates=['Date'], infer_datetime_format=True)
    df = df.drop(['Request-ID', 'Referer-ID'], axis=1)

    # set id's where possible
    df['Request-ID'] = [map_string_to_id(x, page_count_df) for x in df['Request-URI']]
    df['Referer-ID'] = [map_string_to_id(x, page_count_df) for x in df['Referer']]

    # rearrange
    df = df[['Session-ID',
             'Remote-IP',
             'Session-Nr',
             'Date',
             'TimeDelta',
             'Referer-ID',
             'Referer',
             'Request-ID',
             'Request-URI']]
    df.to_csv('logs/big_session_df_remapped.csv', sep='|', header=True, index=False)


def build_coo_transition_matrix():
    """
    Creates .h5 File which holds four arrays so a scipy.sparse Transition Matrix can be created:
        data : Earray
         The Transition Values
        row_indices : Earray
         Row Indices Corresponding to Data Values
        column_indices : Earray
         Column Indices Corresponding to Data Values
        shape_dimensions : Carray
         Dimensions of the Matrix
    """
    # dimension of matrix is the number of pages we indexed
    page_count_df = pd.read_csv('logs/id_name_mapping.csv', sep='|', header=0)
    dimension = len(page_count_df.index)
    del page_count_df

    df_with_page_id = pd.read_csv('logs/big_session_df_remapped.csv', sep='|', header=0)

    # Empty Referers can't be indexed and don't add value(s) to a Transition Matrix
    df_len = len(df_with_page_id.index)
    df_with_page_id = df_with_page_id[~df_with_page_id['Referer-ID'].isnull()]
    # If split_non_af_entries() was called, then there shouldn't be any more rows to drop.
    print "Dropped Rows: " + str(df_len - len(df_with_page_id.index))
    # In case there were still some NaN Values, we cast to int now (not possible with NaN's in Column)
    df_with_page_id['Referer-ID'] = df_with_page_id['Referer-ID'].astype('int64')

    df_with_page_id.set_index(['Referer-ID', 'Request-ID'], inplace=True)
    df_with_page_id = df_with_page_id.sortlevel(level=0, sort_remaining=True)

    # write to h5 file
    with tb.open_file('transition_matrix.h5', 'w') as f:
        filters = tb.Filters(complevel=5, complib='blosc')

        # Earrays
        data = f.create_earray(f.root, 'data', tb.Float32Atom(), shape=(0,), filters=filters)
        row_indices = f.create_earray(f.root, 'row_indices', tb.Float32Atom(), shape=(0,), filters=filters)
        column_indices = f.create_earray(f.root, 'column_indices', tb.Float32Atom(), shape=(0,), filters=filters)
        # Carray
        shape_dimensions = f.create_carray(f.root, 'shape_dimensions', tb.Float32Atom(), shape=(1, 2), filters=filters)
        shape_dimensions[0, :] = dimension

        # Add Values and Indices so the Transition Matrix can be created (from of those)
        for referer_page_id in df_with_page_id.index.levels[0]:
            # Get all Requests with current Referer
            one_page = df_with_page_id.loc[referer_page_id]
            # Sort Targets of that Referer by Count
            out_link_count = one_page.index.value_counts().sort_index()
            # Data of Transition Matrix is the number of Transition from Node A to B, therefore the Link Count
            data.append(out_link_count.values)
            # The Row Index is the current Referer, for each data value (source of Transition)
            row_indices.append([referer_page_id] * len(out_link_count.index))
            # The Targets (of Transition)
            column_indices.append(out_link_count.index.values)


def read_coo_transition_matrix():
    h5 = tb.open_file('transition_matrix.h5', 'r')
    data = h5.root.data
    row_indices = h5.root.row_indices
    column_indices = h5.root.column_indices
    shape_dimensions = h5.root.shape_dimensions
    coo = coo_matrix((data, (row_indices, column_indices)), shape=(shape_dimensions[0, 0], shape_dimensions[0, 1]))
    h5.close()

    """
    # transform to csr for efficient algorithmic operations
    csr = coo.tocsr()
    """
    print "Total sum: " + str(coo.sum())
    print "Diagonal sum: " + str(coo.diagonal().sum())


def print_h5_data_information():
    """
    Print some Information about the Transformation Matrix.
    Creates coo matrix
    """
    h5 = tb.open_file('transition_matrix.h5', 'r')
    print "Last 3 Row Indices: " + str(np.array(h5.root.row_indices)[-3:])
    print "Last 3 Column Indices: " + str(np.array(h5.root.column_indices)[-3:])
    print "Shape: " + str(int(h5.root.shape_dimensions[0, 0])), str(int(h5.root.shape_dimensions[0, 1]))

    print "Trying load Data into coo Matrix...",
    data = h5.root.data
    row_indices = h5.root.row_indices
    column_indices = h5.root.column_indices
    shape_dimensions = h5.root.shape_dimensions
    coo = coo_matrix((data, (row_indices, column_indices)), shape=(shape_dimensions[0, 0], shape_dimensions[0, 1]))
    print " success!"
    h5.close()
    print "Non-Zero Values: " + str(coo.nnz)
    print "Total Matrix Sum: " + str(int(coo.sum()))
    print "Diagonal sum: " + str(int(coo.diagonal().sum()))


def build_future_transition_matrix():
    # The real transition_matrix has ~70.000 non-zero Entries in 50.000^2 fields --> Densitiy 0.000028
    expected_dimension = 150000
    expected_density = 0.001
    # todo: change randint to skewed distribution
    # todo: compare creation speed: coo vs lil
    transition_matrix = rand(m=expected_dimension, n=expected_dimension,
                             density=expected_density, format='coo',
                             random_state=np.random.randint(low=1000))

    # save to h5 file
    filters = tb.Filters(complevel=5, complib='blosc')

    with tb.open_file('future_transition_matrix.h5', 'w') as f:
        # Earrays
        data = f.create_earray(f.root, 'data', tb.Float32Atom(), shape=(0,), filters=filters)
        row_indices = f.create_earray(f.root, 'row_indices', tb.Float32Atom(), shape=(0,), filters=filters)
        column_indices = f.create_earray(f.root, 'column_indices', tb.Float32Atom(), shape=(0,), filters=filters)
        # Carray
        shape_dimensions = f.create_carray(f.root, 'shape_dimensions', tb.Float32Atom(), shape=(1, 2), filters=filters)

        # append values to file
        data.append(transition_matrix.data)
        row_indices.append(transition_matrix.row)
        column_indices.append(transition_matrix.col)

        shape_dimensions[0, 0] = transition_matrix.shape[0]
        shape_dimensions[0, 1] = transition_matrix.shape[1]


def df_information():
    big_df = pd.read_csv('logs/big_session_df_remapped.csv',
                         sep='|',
                         header=0,
                         parse_dates=['Date'],
                         infer_datetime_format=True)
    number_of_requests = len(big_df.index)
    number_of_sessions = len(big_df['Session-Nr'].unique())
    average_session_length = float(number_of_requests) / float(number_of_sessions)
    information_dict = {'Request-Count': number_of_requests,
                        'Session-Count': number_of_sessions,
                        'AvgSessionLength': average_session_length}
    print information_dict.items()

    info_df = pd.DataFrame(columns=['Request-Count', 'Session-Count', 'AvgSessionLength'])
    row_index = len(info_df.index)
    info_df[row_index] = number_of_requests, number_of_sessions, average_session_length
    print info_df


def split_single_click_session_from_big_df_clone():
    big_df = pd.read_csv('logs/one_big_df.csv', sep='|', header=0)
    session_lenghts = big_df['Session-ID'].value_counts()
    session_lenghts = session_lenghts[session_lenghts.values == 1]
    df_mask = big_df['Session-ID'].isin(session_lenghts.index)
    one_click_df = big_df[df_mask]
    big_df = big_df[~df_mask]
    big_df.to_csv('logs/one_big_df.csv', sep='|', header=True, index=False)
    one_click_df.to_csv('logs/one_click_sessions_df.csv', sep='|', header=True, index=False)


def split_non_af_entries_clone():
    """
    Removes all entries where Referer-ID was not set (not Mapped).
    Creates Separate DataFrame for Removed Entries.
    """
    big_df = pd.read_csv('logs/one_big_df.csv', sep='|', header=0)
    df_mask = np.isnan(big_df['Referer-ID'])
    non_af_df = big_df[df_mask]
    big_df = big_df[~df_mask]
    big_df.to_csv('logs/one_big_df.csv', sep='|', header=True, index=False)
    non_af_df.to_csv('logs/non_af_df.csv', sep='|', header=True, index=False)



def main():
    """
    Operations on Daily/Single Files start here
    """
    #remove_milliseconds()
    #load_file_into_df()
    #find_useragent_bots()
    #remove_bot_entries()
    #count_xml_requests(remove_them=False)
    #filter_html_requests()
    #print_jsp_requests()
    #filter_referer()
    #filter_requests_uri()
    #filter_status_codes()
    #filter_servername()
    #filter_admin_requests()
    #filter_af_proxy_ip()  # didn't remove those
    #add_time_delta()
    #remove_non_af_entries()  # deprecated (might need later again)
    """
    Backup of processed folder was created here.
    """
    #count_pages()
    #remove_unneeded_columns()
    #add_page_id()

    #combine_dataframe()
    add_session_number_to_combined_df()
    """
    big_session_df backup was created here.
    """

    #count_all_clicks()
    #count_session_ids()
    #count_remote_ips()
    #count_clicks_from_outside()
    #count_nan_referer()

    """
    Operations on combined DataFrame start here
    """
    #split_single_click_session_from_big_df()
    #split_non_af_entries()
    #filter_long_sessions()
    #remove_trailing_slash()

    #count_pages_in_big_df()
    #create_mapping_for_flo()
    #remap_big_df_page_ids()

    #build_coo_transition_matrix()
    #read_coo_transition_matrix()
    #print_h5_data_information()

    #build_future_transition_matrix()  # not ued (yet?)
    #df_information()  # TODO: Work on that

    #split_single_click_session_from_big_df_clone()
    #split_non_af_entries_clone()

if __name__ == '__main__':
    main()
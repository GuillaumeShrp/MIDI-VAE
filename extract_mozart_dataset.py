import requests, bs4


url = 'https://www.kunstderfuge.com/mozart.htm'
download_url = 'https://www.kunstderfuge.com/'
response = requests.get(url)


parsed = bs4.BeautifulSoup(response.text, 'html.parser')


songs = parsed.find_all("p", {"class": "midi"})

for s in songs:
    print(s.text)
    versions = s.find_all('a')
    
    remove_punctuation_map = dict((ord(char), None) for char in '\/*?:"<>|')
    versions_name = s.text.translate(remove_punctuation_map).replace(" ", "_").split('\n')
    versions_name = [v.split(',')[0] for v in versions_name] #remove song code
    
    if s.text[0] != "»":
        title = versions_name[0]
        versions_name = versions_name[1:] 

    versions_name = [v[8:] for v in versions_name] #remove "midi"

    for ver, name in zip(versions, versions_name):

        link = ver.get('href')
        link_url = requests.compat.urljoin(download_url, link)
        print(link)
        if link != None and link.endswith('.mid'):

            filename = 'data/original/mozart/' + name + '.mid'
            if s.text[0] != "»":
                filename = 'data/original/mozart/' + title + '_' + name + '.mid'
            
            with open(filename, 'wb') as midifile:
                midifile.write(requests.get(link_url).content)
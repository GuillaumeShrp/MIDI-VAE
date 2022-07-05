import requests, bs4, os

save_folder = 'data/complete/classical/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)   

url = "http://www.piano-midi.de/"
parsed = bs4.BeautifulSoup(requests.get(url).text, 'html.parser')
composer_list = parsed.findAll('a', {"class": "navi"})[1:]
composer_list = [c['href'] for c in composer_list]

for c in composer_list:
    download_url = url + c
    parsed = bs4.BeautifulSoup(requests.get(download_url).text, 'html.parser')

    references = parsed.findAll('td', {"class": "midi"})
    for ref in references:
        if ref.find('a'):
            title = ref.find('a')['href']
            if title.endswith('.mid') and not title.endswith('format0.mid'):
                link_url = requests.compat.urljoin(download_url, title)
                filename = save_folder + title.split('/')[2]
                with open(filename, 'wb') as midifile:
                    content = requests.get(link_url).content
                    if len(content) <= 30000: #if midi file size <= 30ko
                        midifile.write(content)
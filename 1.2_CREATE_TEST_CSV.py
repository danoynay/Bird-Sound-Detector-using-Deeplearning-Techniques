import pandas as pd

list_of_labels = [ 
                   'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet',
                   'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet', 'coppersmith_barbet',
                  
                   'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 
                   'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 'black_shama', 
                  
                   'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 
                   'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite', 'brahminy_kite',
                   
                   'brown_hawk_owl', 'brown_hawk_owl',  'brown_hawk_owl', 'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl', 
                   'brown_hawk_owl', 'brown_hawk_owl',  'brown_hawk_owl', 'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl',  'brown_hawk_owl', 
                   
                   'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle',
                   'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle', 'crested_serpent_eagle',
                   
                   'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 
                   'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 'blackish_cuckooshrike', 
                   
                   'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 
                   'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 'orange_bellied_flowerpecker', 
                   
                   'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar',
                   'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar', 'grey_nightjar',
                   
                   'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo',
                   'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo', 'oriental_cuckoo',
                  
                   'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 
                   'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker', 'white_bellied_woodpecker'
                   ]

list_of_audio_fns = [
                     "barbet (1).wav", "barbet (2).wav", "barbet (3).wav", "barbet (4).wav", "barbet (5).wav",
                     "barbet (6).wav", "barbet (7).wav", "barbet (8).wav", "barbet (9).wav", "barbet (10).wav",
                     "barbet (11).wav", "barbet (12).wav", "barbet (13).wav", "barbet (14).wav", "barbet (15).wav",
                     "barbet (16).wav", "barbet (17).wav", "barbet (18).wav", "barbet (19).wav", "barbet (20).wav",

                     "blackshama (1).wav", "blackshama (2).wav", "blackshama (3).wav", "blackshama (4).wav", "blackshama (5).wav",
                     "blackshama (6).wav", "blackshama (7).wav", "blackshama (8).wav", "blackshama (9).wav", "blackshama (10).wav",
                     "blackshama (11).wav", "blackshama (12).wav", "blackshama (13).wav", "blackshama (14).wav", "blackshama (15).wav",
                     "blackshama (16).wav", "blackshama (17).wav", "blackshama (18).wav", "blackshama (19).wav", "blackshama (20).wav",

                     "brahminykite (1).wav", "brahminykite (2).wav", "brahminykite (3).wav", "brahminykite (4).wav", "brahminykite (5).wav",
                     "brahminykite (6).wav", "brahminykite (7).wav", "brahminykite (8).wav", "brahminykite (9).wav", "brahminykite (10).wav",
                     "brahminykite (11).wav", "brahminykite (12).wav", "brahminykite (13).wav", "brahminykite (14).wav", "brahminykite (15).wav",
                     "brahminykite (16).wav", "brahminykite (17).wav", "brahminykite (18).wav", "brahminykite (19).wav", "brahminykite (20).wav",

                     "brownhawkowl (1).wav", "brownhawkowl (2).wav", "brownhawkowl (3).wav", "brownhawkowl (4).wav", "brownhawkowl (5).wav",
                     "brownhawkowl (6).wav", "brownhawkowl (7).wav", "brownhawkowl (8).wav", "brownhawkowl (9).wav", "brownhawkowl (10).wav", 
                     "brownhawkowl (11).wav", "brownhawkowl (12).wav", "brownhawkowl (13).wav", "brownhawkowl (14).wav", "brownhawkowl (15).wav",
                     "brownhawkowl (16).wav", "brownhawkowl (17).wav", "brownhawkowl (18).wav", "brownhawkowl (19).wav", "brownhawkowl (20).wav",

                     "cresentserpent (1).wav", "cresentserpent (2).wav", "cresentserpent (3).wav", "cresentserpent (4).wav", "cresentserpent (5).wav",
                     "cresentserpent (6).wav", "cresentserpent (7).wav", "cresentserpent (8).wav", "cresentserpent (9).wav", "cresentserpent (10).wav",
                     "cresentserpent (11).wav", "cresentserpent (12).wav", "cresentserpent (13).wav", "cresentserpent (14).wav", "cresentserpent (15).wav",
                     "cresentserpent (16).wav", "cresentserpent (17).wav", "cresentserpent (18).wav", "cresentserpent (19).wav", "cresentserpent (20).wav",

                     "cuckooshrike (1).wav", "cuckooshrike (2).wav", "cuckooshrike (3).wav", "cuckooshrike (4).wav", "cuckooshrike (5).wav",
                     "cuckooshrike (6).wav", "cuckooshrike (7).wav", "cuckooshrike (8).wav", "cuckooshrike (9).wav", "cuckooshrike (10).wav",
                     "cuckooshrike (11).wav", "cuckooshrike (12).wav", "cuckooshrike (13).wav", "cuckooshrike (14).wav", "cuckooshrike (15).wav",
                     "cuckooshrike (16).wav", "cuckooshrike (17).wav", "cuckooshrike (18).wav", "cuckooshrike (19).wav", "cuckooshrike (20).wav",

                     "flowerpecker (1).wav", "flowerpecker (2).wav", "flowerpecker (3).wav", "flowerpecker (4).wav", "flowerpecker (5).wav",
                     "flowerpecker (6).wav", "flowerpecker (7).wav", "flowerpecker (8).wav", "flowerpecker (9).wav", "flowerpecker (10).wav",
                     "flowerpecker (11).wav", "flowerpecker (12).wav", "flowerpecker (13).wav", "flowerpecker (14).wav", "flowerpecker (15).wav",
                     "flowerpecker (16).wav", "flowerpecker (17).wav", "flowerpecker (18).wav", "flowerpecker (19).wav", "flowerpecker (20).wav",

                     "greynightjar (1).wav", "greynightjar (2).wav", "greynightjar (3).wav", "greynightjar (4).wav", "greynightjar (5).wav",
                     "greynightjar (6).wav", "greynightjar (7).wav", "greynightjar (8).wav", "greynightjar (9).wav", "greynightjar (10).wav",
                     "greynightjar (11).wav", "greynightjar (12).wav", "greynightjar (13).wav", "greynightjar (14).wav", "greynightjar (15).wav",
                     "greynightjar (16).wav", "greynightjar (17).wav", "greynightjar (18).wav", "greynightjar (19).wav", "greynightjar (20).wav",

                     "orientalcuckoo (1).wav", "orientalcuckoo (2).wav", "orientalcuckoo (3).wav", "orientalcuckoo (4).wav", "orientalcuckoo (5).wav",
                     "orientalcuckoo (6).wav", "orientalcuckoo (7).wav", "orientalcuckoo (8).wav", "orientalcuckoo (9).wav", "orientalcuckoo (10).wav",
                     "orientalcuckoo (11).wav", "orientalcuckoo (12).wav", "orientalcuckoo (13).wav", "orientalcuckoo (14).wav", "orientalcuckoo (15).wav",
                     "orientalcuckoo (16).wav", "orientalcuckoo (17).wav", "orientalcuckoo (18).wav", "orientalcuckoo (19).wav", "orientalcuckoo (20).wav",

                     "woodpecker (1).wav", "woodpecker (2).wav", "woodpecker (3).wav", "woodpecker (4).wav", "woodpecker (5).wav",
                     "woodpecker (6).wav", "woodpecker (7).wav", "woodpecker (8).wav", "woodpecker (9).wav", "woodpecker (10).wav",
                     "woodpecker (11).wav", "woodpecker (12).wav", "woodpecker (13).wav", "woodpecker (14).wav", "woodpecker (15).wav",
                     "woodpecker (16).wav", "woodpecker (17).wav", "woodpecker (18).wav", "woodpecker (19).wav", "woodpecker (20).wav"
                    ]
                   
print(len(list_of_labels), len(list_of_audio_fns))
X = []

for fn, label in zip(list_of_audio_fns, list_of_labels):
    X.append([fn, label])

df = pd.DataFrame(X)
df.columns = ['filename', 'label']
df.to_csv('TEST_BIRDS.csv', index = False)
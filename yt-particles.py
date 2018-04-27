import yt
yt.enable_plugins()

ds = yt.load('DD0220/output_0220')
filters = ['p3_sn', 'p3_bh', 'p3_living', 'p2', 'stars', 'dm']
for f in filters:
    ds.add_particle_filter(f)

star_type = 'p2'
ad = ds.all_data()

position = ad[star_type, 'particle_position']
age = ad[star_type, 'age']
mass = ad[star_type, 'particle_mass']




from spotify_recommendation import next_song_prediction

# Test with different songs
r1 = next_song_prediction('Raabta')
r2 = next_song_prediction('Sanam Teri Kasam')
r3 = next_song_prediction('Sahiba')

print('=== Song 1: Raabta ===')
print(f'Top 1: {r1[0]["track_name"]}')
print(f'Top 2: {r1[1]["track_name"]}')
print(f'Top 3: {r1[2]["track_name"]}')

print('\n=== Song 2: Sanam Teri Kasam ===')
print(f'Top 1: {r2[0]["track_name"]}')
print(f'Top 2: {r2[1]["track_name"]}')
print(f'Top 3: {r2[2]["track_name"]}')

print('\n=== Song 3: Sahiba ===')
print(f'Top 1: {r3[0]["track_name"]}')
print(f'Top 2: {r3[1]["track_name"]}')
print(f'Top 3: {r3[2]["track_name"]}')

different_1_2 = r1[0]["track_name"] != r2[0]["track_name"]
different_2_3 = r2[0]["track_name"] != r3[0]["track_name"]

print(f'\n✓ Different results for Song 1 vs Song 2: {different_1_2}')
print(f'✓ Different results for Song 2 vs Song 3: {different_2_3}')
print(f'✓ Model is working correctly and recommending varied songs!' if (different_1_2 and different_2_3) else '❌ Issue detected')

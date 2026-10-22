# Notes
## Day 1

I live in Almaty, and the air here gets really bad in winter. That's why I wanted to do a project about local air pollution.

CHP-2 burns coal to produce electricity and heat. It is being converted to gas, and the plan is to finish the work by the end of 2026. This should reduce pollution from the plant. I want to see how much difference it makes to the city's air. Weather matters too, so if the air gets cleaner, part of the change could be because of the weather.

My first goal is to predict tomorrow's average PM2.5 using machine learning. Later, I want to compare the data before and after the switch to gas, taking weather into account.

I created the repository and added scripts to download weather data, CAMS forecasts and PM2.5 measurements from the US Embassy station through OpenAQ.

## Day 2

I ran all three download scripts. The data is now on my computer.

- Weather: 53,472 hourly rows, about 6 years of data.
- CAMS forecast: 36,816 hourly rows, starting in August 2022.
- US Embassy PM2.5: 37,822 hourly rows, from April 2020 to November 2025.

There are no embassy PM2.5 rows for 2026. The station stopped working on 14 November 2025, so this source won't cover that year.

The median hourly PM2.5 was 15 µg/m³, and the average was about 30. The highest hourly value was 421 µg/m³ on 19 January 2021. The WHO guideline is 15 µg/m³ for a daily average. I need to calculate daily averages to compare the measurements with it.

I plan to use the available data from 2020-2025 for training. CAMS will only cover the part from August 2022 onward. For the 2026 CHP-2 analysis and demo, I need another PM2.5 source. I might use AirGradient sensors in Almaty.

I left the data files out of GitHub because they're large. Anyone can download them by running the scripts.

## Day 3

I used `build_dataset.py` to put weather, CAMS and PM2.5 data into one hourly table. I kept hours with missing data too, so the gaps are easy to see. I also removed PM2.5 values below 0 or above 1000 µg/m³.

About 34% of the PM2.5 data is missing. In 2021-2022, around 45% is missing, but in 2024 it's only about 5%. There are no embassy measurements for 2026 because the station had already stopped working.

I expected PM2.5 to be worst in the evening, but the average winter daily cycle peaks at 15:00. The lowest point is at 06:00. I checked nearby Clarity sensors, and they showed the same pattern.

Maybe daytime winds carry polluted air uphill from the lower parts of the city. I haven't confirmed that explanation yet.

## Day 4

I made the first plots in `01_eda.ipynb`. Winter PM2.5 is about 6 times higher than in summer. The average is around 70 µg/m³ in January and 67 in December, compared with 10-12 in summer.

During the heating season, weekdays and weekends look almost the same. Both average around 50-55 µg/m³. Traffic might not be the main cause of winter smog, but this comparison doesn't prove it.

The 10 most polluted days were all in December or January, between 2020 and 2023. Daily averages were 170-222 µg/m³. Temperatures were between -14 and -6 °C, and the boundary layer was only about 25-70 m high. With less air to mix into, pollution could build up near the ground.

The wind wasn't always weak on those days. Cold weather and poor air mixing may explain more than wind speed alone.

## Day 5

I compared daily PM2.5 with the weather. The strongest correlation was with temperature at -0.65, followed by boundary layer height, or BLH, at -0.51. Wind speed had almost no correlation at 0.03.

BLH tells us how high the air can mix. When it's low, pollution can stay trapped near the ground. The median line on the plot also shows lower PM2.5 as BLH gets higher.

I also checked what happened around the start of heating, roughly 15 October. PM2.5 went from 25 to 31 µg/m³ in 2020, from 14 to 20 in 2023, and from 18 to 31 in 2025. The change was small in 2022 and 2024. There wasn't enough data around this period for the same comparison in 2021.

PM2.5 often went up around the start of heating. It also gets colder around then, so the increase may not be from heating alone.

## Day 6

I compared CAMS forecasts with measured PM2.5. In winter, from December to February, CAMS averaged about 34 µg/m³ while the measurements averaged 61. The bias was -27 µg/m³, and the mean absolute error, or MAE, was about 40.

Most winter points on the scatter plot were below the red line. CAMS often predicted much lower PM2.5 than was actually measured.

In summer, the forecasts were much closer. The MAE was about 6 µg/m³. CAMS was a little too high on average, with a bias of +3 µg/m³.

The biggest errors were in winter, when Almaty has its worst smog. That's where I want my model to improve the forecasts most.

## Day 7

I wrote what I learned in the notebook and updated the README. I added two things about the data: the time zone change in March 2024 and missing BLH data for January-June 2024. I also added two plots to the README: PM2.5 by month and the daily cycle.

## Day 8

I made features.py to turn hourly data into one row per day with 38 features. The model predicts tomorrow's average PM2.5 using data available by midnight. The strongest feature is PM2.5 at 23:00 today, which makes sense because bad air late at night often carries over into the next day.

Tomorrow's weather comes from actual recorded data, so it works like a perfect forecast. Real forecasts can be wrong, so the results may look better than they would in practice.
## Day 9

I made evaluate.py. It calculates MAE, RMSE and how often the model gets the air quality category right (low, high or very high).

I split the data by time: train is September 2020 - 2023, validation is 2024 and test is 2025. I can't shuffle the days, because then the model would see days from the future and the results would look better than they really are.

At first I wanted to test on 2026, but the embassy station stopped working on 14 November 2025. So the test year is 2025.

The simplest baseline is "tomorrow = today". It got an MAE of 8.83 µg/m³ on the test year. CAMS got 10.19, so even this simple guess is more accurate than CAMS for Almaty.
## Day 10

I made train.py and compared four forecasts on the 2025 test data. CAMS got MAE 10.19 µg/m³, linear regression 9.55, the baseline 8.83 and LightGBM 6.84.

LightGBM did best in this run. Its MAE was about 33% lower than CAMS and 23% lower than the baseline. It also got the PM2.5 group right on about 92% of test days.

Linear regression did worse than just saying "tomorrow = today". Maybe it misses how cold weather, wind and heating work together. The scores alone don't tell me why it did worse.

I used early stopping to choose how many trees to use. It stops training when the validation score stops improving, which helps reduce overfitting. Then I trained LightGBM again on train + validation with the same number of trees, so the final model could also use 2024, the year just before the test.
 ## Day 11

  I checked the 10 biggest LightGBM errors. All were forecasts made in January, February and early March 2025. On all 10, the prediction was too low. Measured PM2.5 was around 80-125 µg/m³, while the model predicted 40-80.

  The weather was similar in these cases: about -9 to -2 °C, weak wind and very low BLH, around 30-80 m. There was almost no rain, and only one of the days was a holiday (7 January). The model predicted bad air, but the actual smog was much worse.

  In February the average error was about 21 µg/m³, in January about 15, and in summer only 1-4. Sudden jumps in PM2.5 from one day to the next were also hard for the model.

  I think there may be too few extreme days in the training data. I also used MAE, which doesn't penalise big errors as strongly as squared error does. These could be part of the reason it misses the highest peaks. CAMS was even lower on these days. In February it predicted about 40 µg/m³ when the measurements were 110-125.

  I also tried removing the features that use tomorrow's recorded weather. MAE went from 6.84 to 7.11 µg/m³. Without those features, the MAE was still about 30% lower than CAMS.

  The dates in the error table show when each forecast was made. Each prediction is for the following day
## Day 12

Today I used SHAP to better understand how my LightGBM model makes predictions. It helped me see which features matter the most and how they affect the predicted pollution level on different days.

I divided the features into four groups: weather, heating/season, traffic (weekdays and holidays), and past pollution levels. After leaving out past pollution from the SHAP summary, weather made up about 92% of the remaining importance, heating/season about 7%, and traffic almost 0%. These numbers describe how my model uses the features, not what causes the pollution. They do not mean that weather causes 92% of the smog. For traffic, I only used weekdays, weekends and holidays, with no actual traffic measurements. So the low importance does not mean cars have no effect.

I also checked the three days with the highest pollution levels. All three were very cold (-6 to -14°C), and the boundary layer height (BLH) was only around 24–34 meters. This suggests that the weather made it harder for pollution to spread out.

One problem is that weather and heating are closely connected. When it gets colder, people use more coal for heating, so it's difficult to tell how much each one contributes. I also didn't have actual traffic data, only weekdays and holidays as indicators, so I can't really draw conclusions about the impact of cars.
## Day 13

Today I made a model that only uses weather and calendar features. I left out past PM2.5 because after the switch to gas, those values could already include the change I want to measure. The model could just follow the lower pollution instead of showing how different it is from before.

I tested it by leaving one winter out, training on the other years and predicting the hidden winter. I repeated this for five winters. I compared the average PM2.5 over each heating season. Three of the five had gaps of about 8-12%.

The worst result was 2020-21. The model expected an average of 49.6 µg/m³, but the measured average was 67.1. That was 35.4% higher than expected. I still need to check why it missed that winter so badly.

For 2023-24, BLH is missing for part of the heating season. The measured PM2.5 was 22.5% below the model's estimate. When I removed the BLH features, the gap fell to 13.3%. Missing BLH could be part of the problem, but removing it made the other winters worse.

If PM2.5 after the CHP-2 switch is about 10% below what the model expects, that could still be a model error. I already saw bigger gaps in the old data. Even a larger drop wouldn't prove on its own that it came from the switch to gas.
## Day 14

Today I made a tracker for CHP-2. It compares measured PM2.5 after the switch to gas with what the weather model expects for the same conditions, based on the coal years.

I can't just compare the winters before and after. One winter could be warmer, windier or have better air mixing. Other pollution sources could change too.

In the past winter checks, the gap between measured and predicted seasonal averages was as big as 35.4%. So a drop after the switch could still be a model error. I can't use 35.4% as a strict cutoff either. Even a bigger difference wouldn't prove that CHP-2 caused it.

There are no measurements after the switch yet. I plan to run the tracker once a week when new PM2.5 data becomes available.
## Day 15

Today I made a simple demo with Streamlit. It shows the model's PM2.5 forecast for the next day, the level (low, high or very high) and charts for the last 14 days and the last week.

The weather archive is about 5 days behind. For the recent days and tomorrow, I use weather data from the Open-Meteo forecast API.

The embassy station stopped sending data in November 2025, so the demo still shows a forecast for an old date. I need to connect the AirGradient sensors so it can use current PM2.5 measurements.

For now it only runs on my computer. I might put it online later, once it works with current data.
## Day 16

I got feedback on my project and fixed a few things.

My SHAP explanation was misleading. After leaving past pollution out of the summary, weather had about 92% of the remaining importance. This shows how the model uses the features, not what causes the smog. I also clarified that my traffic features are just weekdays, weekends and holidays. Their low importance does not mean cars have no effect.

I renamed the air groups from good / unhealthy / very unhealthy to low / high / very high in the code, app and notes. My lowest group went up to 35.4 µg/m³, so calling it "good" was misleading (on the EPA scale, "good" ends at 9). These are now labelled as my own concentration groups.

For the test comparison, I retrained linear regression on train + validation, just like LightGBM. Its MAE improved slightly, from 9.66 to 9.55. LightGBM still performed better, with an MAE of 6.84.

Next, I want to test the model with weather forecasts instead of actual next-day weather and look more closely at the large errors on polluted days. The CHP-2 analysis still needs data from after the switch.
## Day 17 — AirGradient

The embassy station stopped reporting on 14 November 2025. I needed another source of PM2.5 data to keep the project going, so I looked at nearby AirGradient sensors.

The main problem was that their readings were higher than the embassy measurements. Only one sensor had enough overlapping data for a comparison. Across 331 shared days, it read about 1.45 times higher, so I used a correction factor of 0.69. The correlation between the two sources was 0.93, but that alone does not mean their readings agree.

I applied this correction to seven sensors within 1.6 km of the station coordinates listed in OpenAQ. I took their median so one unusual reading would have less effect. The correction comes from just one sensor, so applying it to all seven is still an assumption. I also need to verify the station coordinates.

On the AirGradient days, LightGBM got an MAE of 6.98 µg/m³, compared with 7.15 for the “tomorrow = today” baseline. That is only a small improvement. I kept these results separate from the original test because the data source had changed.

## Day 18 — Telegram bot

I made a Telegram bot that sends the PM2.5 forecast for tomorrow. It runs through GitHub Actions at 22:00 Almaty time, so I don’t have to leave my computer on. I put the API keys in GitHub Secrets.

Getting it to run there took a few fixes. Everything worked on my computer, but GitHub started without the data files I already had locally. The training code failed when the AirGradient table was empty. I hadn’t caught that on my own machine.

Even after I fixed it, I kept getting the same error. I was clicking “Re-run”, which used the old commit. Once I started a new run, it picked up the fix.

I also noticed a problem with the message itself. At 26 µg/m³, the bot showed a green icon and said “low”. My dad could read that as “the air is good”, even though it falls in the EPA’s moderate category. I changed the bot to use the full EPA categories and added a short recommendation.
## Day 19: AirCast weights

Some days were especially dirty, but the model didn't pay much attention to them. To keep the average error low, it pulled its predictions toward normal days and underpredicted heavy smog. In many tasks that isn't critical, but with air quality it can be harmful, because those are the days when people need a warning the most.

I read the AirCast paper from MBZUAI. They had the same problem on a much bigger scale and solved it by giving rare days more weight. I did something similar. I put the days into 10 µg/m³ bins by PM2.5 level and counted how common each bin is. Days from rare bins get a bigger penalty for errors, so the model can't ignore them.

I also had to choose how strong this penalty should be. With a strong penalty the test results looked good, but on validation the model started to predict about 12 µg/m³ too high. So I chose by validation, not by test, and kept the penalty weak: days above 100 µg/m³ count about 1.3 times more than normal days, and the rarest ones about 1.6 times.

On the test set the error went down from 6.84 to 6.39, and on the 10% dirtiest days from 24.9 to 19.5. On the AirGradient days it went from 6.98 to 6.34. On validation the overall error got slightly worse, from 8.87 to 9.03, but on the dirtiest days it improved from 15.95 to 14.84.

So the model now underpredicts real pollution less. The Telegram bot uses the new model, and the demo app still uses the old one.
## Day 21 — The 23:00 problem in the bot

The most important feature in my model is PM2.5 at 23:00, the last hour of the day. In training this value was always there. But the bot runs at 22:00, so at that moment 23:00 hasn't happened yet, and the value was empty in every live forecast.

The model still gave a number, so I didn't see any error. I only noticed it when I checked which data the bot actually gets.

I tested how bad it was: on the test year, with this value empty, the error would go from 6.39 to 8.72. That's almost the same as just saying "tomorrow = today", so the model was losing most of its advantage.

Now the bot takes the last hour that is already measured (usually 21:00) instead of 23:00. With this, the error on the test year is 6.34, almost the same as before.

This only changes the live bot. The test results and figures in the README were not affected, because there all the data was already there

## Day 20 — Weather timeout

Today the bot didn't send the forecast. The log showed that the problem wasn't in my code: the weather site (Open-Meteo) didn't answer in time. Before, the bot asked only once and waited 30 seconds, and if there was no answer, it stopped.

Now it tries 3 times, waits up to 60 seconds each time, and pauses 30 seconds between tries. It only fails if all three tries don't work.
## Day 22 — BLH × wind (idea from a paper)

I read Tursumbayeva et al. (2022). They used weather balloon data for Almaty and found that the worst smog days are calm days with a low boundary layer. They multiply the two, BLH × wind speed, and call it the ventilation coefficient: if it's small, the smog stays over the city.

My model had BLH and wind as separate features, but not their product. So I added it, and tried three versions: for today, for tomorrow, and for tomorrow night.

It didn't help. On validation the error got worse in every version, for example 9.03 → 9.63 with tomorrow's BLH × wind. On the test some numbers went a bit up and some a bit down, but I choose by validation, so I didn't add the feature.

I think the reason is that LightGBM is made of trees, and a tree can already combine the two: first split by BLH, then by wind. So the product didn't give it anything new.

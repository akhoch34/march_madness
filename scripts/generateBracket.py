from bracketeer import build_bracket

b = build_bracket(
    outputPath="../data/2021_bracket_test.png",
    teamsPath="../data/MTeams.csv",
    seedsPath="../data/MNCAATourneySeeds.csv",
    slotsPath="../data/MNCAATourneySlots.csv",
    submissionPath="../data/ncaa-march-madness-submission-test.csv",
    year=2021
)
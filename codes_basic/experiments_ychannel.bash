for scale in 2 3 4
do
    for wavelets in False
    do
        echo "Starting net with scale: $scale, wavelets: $wavelets, y-channel: True"
        python main.py --scale $scale --wavelets $wavelets --y-channel True --nEpochs 50
        echo ""
    done
done

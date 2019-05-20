for scale in 2 3 4
do
    for wavelets in False True
    do
        for lchannel in False True
        do
            echo "Starting net with scale: $scale, wavelets: $wavelets, l-channel: $lchannel"
            python main.py --scale $scale --wavelets $wavelets --l-channel $lchannel
            echo ""
        done
    done
done

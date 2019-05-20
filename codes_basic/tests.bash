for dataset in BSDS100 Manga109 Set14 Set5 Urban100 valid
do
    for scale in 2 3 4
    do
        for wavelets in False True
        do
            for lchannel in False True
            do
                echo "Starting validation on $dataset with scale: $scale, wavelets: $wavelets, l-channel: $lchannel"
                python validate.py --dataset $dataset --scale $scale --wavelets $wavelets --l-channel $lchannel
                echo ""
            done
        done
    done
done
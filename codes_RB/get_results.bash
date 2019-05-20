for scale in 2 3 4
do
    for wavelets in False True
    do
            echo "Starting validation with scale: $scale, wavelets: $wavelets"
            python find_best_epoch.py --scale $scale --wavelets $wavelets --y-channel True
            echo ""
    done
done

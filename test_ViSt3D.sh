python inference.py --content_video \
                     data/content\ videos/content28.mp4 \
                     --style_path data/style\ images/45629.jpg \
                     --decoder checkpoints/decoder_iter_40000.pth.tar \
                     --disentangle_appearance checkpoints/disentangle_appearance_iter_5000.pth.tar \
--entangle_feats checkpoints/entangle_feats_iter_100000.pth.tar \
--c3d checkpoints/encoder_iter_10000.pth.tar \
--reverse_clip \
--output ./output

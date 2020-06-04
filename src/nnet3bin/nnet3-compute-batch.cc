// nnet3bin/nnet3-compute-batch.cc

// Copyright 2012-2018   Johns Hopkins University (author: Daniel Povey)
//           2018        Hang Lyu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-batch-compute.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Propagate the features through raw neural network model "
        "and write the output.  This version is optimized for GPU use. "
        "If --apply-exp=true, apply the Exp() function to the output "
        "before writing it out.\n"
        "\n"
        "Usage: nnet3-compute-batch [options] <nnet-in> <features-rspecifier> "
        "<matrix-wspecifier>\n"
        " e.g.: nnet3-compute-batch final.raw scp:feats.scp "
        "ark:nnet_prediction.ark\n";

    ParseOptions po(usage);
    Timer timer;

    NnetBatchComputerOptions opts;
    opts.acoustic_scale = 1.0;  // by default do no scaling

    bool apply_exp = false, use_priors = false;
    std::string use_gpu = "yes";

    std::string word_syms_filename;
    std::string ivector_rspecifier,
                online_ivector_rspecifier,
                utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    opts.Register(&po);

    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per "
                "utterance by default, or per speaker if you provide the "
                "--utt2spk option.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "utt2spk option used to get ivectors per speaker");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("apply-exp", &apply_exp, "If true, apply exp function to "
                "output");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("use-priors", &use_priors, "If true, subtract the logs of the "
                "priors stored with the model (in this case, "
                "a .mdl file is expected as input).");

#if HAVE_CUDA==1
    CuDevice::RegisterDeviceOptions(&po);
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().AllowMultithreading();
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                matrix_wspecifier = po.GetArg(3);

    Nnet raw_nnet;
    AmNnetSimple am_nnet;
    if (use_priors) {
      bool binary;
      TransitionModel trans_model;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &raw_nnet);
    }
    Nnet &nnet = (use_priors ? am_nnet.GetNnet() : raw_nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    Vector<BaseFloat> priors;
    if (use_priors)
      priors = am_nnet.Priors();

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);

    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    int32 num_success = 0, num_fail = 0;
    std::string output_uttid;
    Matrix<BaseFloat> output_matrix;


    NnetBatchInference inference(opts, nnet, priors);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features = feature_reader.Value();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      const Matrix<BaseFloat> *online_ivectors = NULL;
      const Vector<BaseFloat> *ivector = NULL;
      if (!ivector_rspecifier.empty()) {
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          ivector = new Vector<BaseFloat>(ivector_reader.Value(utt));
        }
      }
      if (!online_ivector_rspecifier.empty()) {
        if (!online_ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No online iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          online_ivectors = new Matrix<BaseFloat>(
              online_ivector_reader.Value(utt));
        }
      }

      inference.AcceptInput(utt, features, ivector, online_ivectors,
                            online_ivector_period);

      std::string output_key;
      Matrix<BaseFloat> output;
      while (inference.GetOutput(&output_key, &output)) {
        if (apply_exp)
          output.ApplyExp();
        matrix_writer.Write(output_key, output);
        num_success++;
      }
    }

    inference.Finished();
    std::string output_key;
    Matrix<BaseFloat> output;
    while (inference.GetOutput(&output_key, &output)) {
      if (apply_exp)
        output.ApplyExp();
      matrix_writer.Write(output_key, output);
      num_success++;
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed << "s";
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (num_success != 0) {
      return 0;
    } else {
      return 1;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

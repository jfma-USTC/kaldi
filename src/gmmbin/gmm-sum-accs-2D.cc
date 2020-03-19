// gmmbin/gmm-sum-accs-2D.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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

#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
//#include "hmm/transition-model.h"

// ����ɢ�� JOB ���ļ��С���gmm-acc-stats-ali-2D ���ɵ��ۼ����� ��Ӧ��ͬ trans-id��pdf-id ���ۼ����ϲ���һ��
int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for GMM training.\n"
        "Usage: gmm-sum-accs-2D [options] <stats-out> <stats-in1> <stats-in2> ...\n"
        "E.g.: gmm-sum-accs-2D 1.acc 1.1.acc 1.2.acc\n";

    bool binary = true;
    kaldi::ParseOptions po(usage); // ʹ��usage�ִ���ʼ��һ��ParseOptions���ʵ��po
    po.Register("binary", &binary, "Write output in binary mode"); // ��ParseOptions����ע��������ѡ��(Option�Ľṹ�������Լ���ע�ắ��)
	po.Read(argc, argv); // �������в������н���
	// ����Ƿ������Ч������λ�ò���
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

	// ��ȡָ��λ�õ������в���������ֵ����Ӧ��ѡ��
    std::string stats_out_filename = po.GetArg(1); // �ϲ��ۻ���������ļ���
	kaldi::Vector<double> transition_accs_top_down;
	kaldi::Vector<double> transition_accs_left_right;
    kaldi::AccumAmDiagGmm gmm_accs;

    int num_accs = po.NumArgs() - 1;
    for (int i = 2, max = po.NumArgs(); i <= max; i++) { // ��ȡ����������ۻ����ļ����ۼ�
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
	  transition_accs_top_down.Read(ki.Stream(), binary_read, true /*add read values*/);
	  transition_accs_left_right.Read(ki.Stream(), binary_read, true /*add read values*/);
      gmm_accs.Read(ki.Stream(), binary_read, true /*add read values*/); // trueѡ���ʾ�ۼ�
    }

	// ����ۻ��������������־��Ϣ
    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
	  transition_accs_top_down.Write(ko.Stream(), binary);
	  transition_accs_left_right.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Summed " << num_accs << " stats, total count "
              << gmm_accs.TotCount() << ", avg like/frame "
              << (gmm_accs.TotLogLike() / gmm_accs.TotCount());
    KALDI_LOG << "Total count of stats is " << gmm_accs.TotStatsCount();
    KALDI_LOG << "Written stats to " << stats_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



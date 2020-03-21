// gmmbin/gmm-est.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "gmm/am-diag-gmm.h"
//#include "tree/context-dep.h"
#include "hmm/transition-model-2D.h"
#include "gmm/mle-am-diag-gmm.h"

// ��Ҫ�������֣�һ���ָ��� TransitionModel��һ���ָ���GMM
int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;

		const char *usage =
			"Do Maximum Likelihood re-estimation of GMM-based acoustic model\n"
			"Usage:  gmm-est-2D [options] <model-in> <stats-in> <model-out>\n"
			"e.g.: gmm-est 1.mdl 1.acc 2.mdl\n";

		bool binary_write = true;
		MleTransitionUpdateConfig tcfg;
		MleDiagGmmOptions gmm_opts;
		int32 mixup = 0;
		int32 mixdown = 0;
		BaseFloat perturb_factor = 0.01;
		BaseFloat power = 0.2;
		BaseFloat min_count = 20.0;
		std::string update_flags_str = "mvwt";
		std::string occs_out_filename;

		ParseOptions po(usage); // ʹ��usage�ִ���ʼ��һ��ParseOptions���ʵ��po
		// ��ParseOptions����ע��������ѡ��(Option�Ľṹ�������Լ���ע�ắ��)
		po.Register("binary", &binary_write, "Write output in binary mode");
		po.Register("mix-up", &mixup, "Increase number of mixture components to "
			"this overall target."); // mix-upѡ��ָ����ǰ��Ҫ�ﵽ�ĸ�˹��
		po.Register("min-count", &min_count,
			"Minimum per-Gaussian count enforced while mixing up and down.");
		po.Register("mix-down", &mixdown, "If nonzero, merge mixture components to this "
			"target.");
		po.Register("power", &power, "If mixing up, power to allocate Gaussians to"
			" states.");
		po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
			"update: subset of mvwt.");
		po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
			"means by standard deviation times this factor.");
		po.Register("write-occs", &occs_out_filename, "File to write pdf "
			"occupation counts to.");
		tcfg.Register(&po);
		gmm_opts.Register(&po);

		po.Read(argc, argv); // �������в������н���
		// ����Ƿ������Ч������λ�ò���
		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}

		kaldi::GmmFlagsType update_flags =
			StringToGmmFlags(update_flags_str); // update_flags_str���'mvwt'���Ӵ���ָ����ЩGMM������Ҫ����

		// ��ȡָ��λ�õ������в���������ֵ����Ӧ��ѡ��
		std::string model_in_filename = po.GetArg(1),
			stats_filename = po.GetArg(2),
			model_out_filename = po.GetArg(3);

		// ����һ��ģ�ͣ�.mdl���ж�ȡTM��GMMs����Ϣ��trans_model��am_gmm
		AmDiagGmm am_gmm;
		TransitionModel_2D trans_model;
		{
			bool binary_read;
			Input ki(model_in_filename, &binary_read);
			trans_model.Read(ki.Stream(), binary_read);
			am_gmm.Read(ki.Stream(), binary_read);
		}

		// ���ۻ����ļ��ж�ȡ
		// ��1�������ļ�����frame�г��ֵ�trans_id����ֵ��transition_accs��
		// ��2��ÿ��DiagGmm���������ͳ������gmm_accs��
		Vector<double> transition_accs_top_down;
		Vector<double> transition_accs_left_right;
		AccumAmDiagGmm gmm_accs;
		{
			bool binary;
			Input ki(stats_filename, &binary);
			transition_accs_top_down.Read(ki.Stream(), binary);
			transition_accs_left_right.Read(ki.Stream(), binary);
			gmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
		}

		if (update_flags & kGmmTransitions) {  // Update transition model.
			BaseFloat objf_impr, count;
			// MleUpdate����trans_id��ͳ����������trans_id��log_prob_����¼trans_id���ܸ�����count����Լ����frame������Ϊframe��frame����״̬ת�ƣ����Լ�
			// objf_impr = counts(tid) * (Log(new_probs(tid)) - Log(old_probs(tid))) Ϊÿ��ת�ƻ����ִ���������ת�Ƹ��ʵĶ�����˻�֮��
			trans_model.MleUpdate_TopDown(transition_accs_top_down, tcfg, &objf_impr, &count);
			KALDI_LOG << "Transition model update: Overall " << (objf_impr / count)
				<< " log-like improvement per frame over " << (count)
				<< " frames. In TopDown direction.";
			trans_model.MleUpdate_LeftRight(transition_accs_left_right, tcfg, &objf_impr, &count);
			KALDI_LOG << "Transition model update: Overall " << (objf_impr / count)
				<< " log-like improvement per frame over " << (count)
				<< " frames. In LeftRight direction.";
		}

		{  // Update GMMs.
			BaseFloat objf_impr, count;
			BaseFloat tot_like = gmm_accs.TotLogLike(),
				tot_t = gmm_accs.TotCount();
			MleAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
				&objf_impr, &count);
			KALDI_LOG << "GMM update: Overall " << (objf_impr / count)
				<< " objective function improvement per frame over "
				<< count << " frames";
			KALDI_LOG << "GMM update: Overall avg like per frame = "
				<< (tot_like / tot_t) << " over " << tot_t << " frames.";
		}

		if (mixup != 0 || mixdown != 0 || !occs_out_filename.empty()) {
			// get pdf occupation counts
			Vector<BaseFloat> pdf_occs;
			pdf_occs.Resize(gmm_accs.NumAccs());
			for (int i = 0; i < gmm_accs.NumAccs(); i++)
				pdf_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum(); // p(m|Oj)��j��m��ͣ�����OjΪ�����GMM���ɵĹ۲�������֡����m��ʾ���GMM�е�m������

			if (mixdown != 0)
				am_gmm.MergeByCount(pdf_occs, mixdown, power, min_count);

			if (mixup != 0)
				am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor,
					power, min_count);

			if (!occs_out_filename.empty()) {
				bool binary = false;
				WriteKaldiObject(pdf_occs, occs_out_filename, binary);
			}
		}

		{
			Output ko(model_out_filename, binary_write);
			trans_model.Write(ko.Stream(), binary_write);
			am_gmm.Write(ko.Stream(), binary_write);
		}

		KALDI_LOG << "Written model to " << model_out_filename;
		return 0;
	}
	catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}
}



// gmmbin/gmm-init-mono-2D.cc

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
#include "hmm/hmm-topology-2D.h"
#include "hmm/transition-model-2D.h"
#include "tree/context-dep.h"

namespace kaldi {
	// This function reads a file like:
	// 1 2 3
	// 4 5
	// 6 7 8
	// where each line is a list of integer id's of phones (that should have their pdfs shared).
	/*
	void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
		list_out->clear(); // list_outΪ��ά����ָ�룬�˴�������clear������listoutָ��Ķ�ά�������
		Input input(rxfilename);
		std::istream &is = input.Stream();
		std::string line;
		while (std::getline(is, line)) {
			list_out->push_back(std::vector<int32>()); // �ڶ�ά�������׷��һ��������
			if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back()))) // ��\t��\r�ָ�line����ֵ����׷�ӵĿ�����
				KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
				<< PrintableRxfilename(rxfilename) << ")";
			std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
			if (!IsSortedAndUniq(*(list_out->rbegin())))
				KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
				<< " (reading " << PrintableRxfilename(rxfilename) << ")";
		}
	}
	*/

} // end namespace kaldi

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		using kaldi::int32;

		const char *usage =
			"Initialize monophone GMM.\n"
			"Usage:  gmm-init-mono-2D <topology-in> <dim> <model-out> \n"
			"e.g.: \n"
			" gmm-init-mono-2D topo 39 mono.mdl\n";
		// ����ѡ�����Ĭ��ֵ
		bool binary = true;
		std::string train_feats;
		std::string shared_phones_rxfilename;
		BaseFloat perturb_factor = 0.0;
		// ʹ��usage�ִ���ʼ��һ��ParseOptions���ʵ��po
		ParseOptions po(usage);
		// ��ParseOptions����ע��������ѡ��(Option�Ľṹ�������Լ���ע�ắ��)
		// ����ο���https://shiweipku.gitbooks.io/chinese-doc-of-kaldi/content/parsing_cmd_options.html
		po.Register("binary", &binary, "Write output in binary mode");
		po.Register("train-feats", &train_feats,
			"rspecifier for training features [used to set mean and variance]");
		po.Register("shared-phones", &shared_phones_rxfilename,
			"rxfilename containing, on each line, a list of phones whose pdfs should be shared.");
		po.Register("perturb-factor", &perturb_factor,
			"Perturb the means using this fraction of standard deviation.");
		// �������в������н���
		po.Read(argc, argv);
		// ����Ƿ������Ч������λ�ò���
		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}

		// The positional arguments get read here (they can only be obtained
		// from ParseOptions as strings).
		// ��ȡָ��λ�õ������в���������ֵ����Ӧ��ѡ��
		// ��ע����Щ����ֻ��ͨ��ParseOptions�����ַ�������ʽ��ȡ��
		std::string topo_filename = po.GetArg(1);
		int dim = atoi(po.GetArg(2).c_str());  // atoi���ַ���ת����������
		// ��c_str()������string��ת��Ϊchar *���ͣ�����atoi��char *����ת��Ϊ���β���ֵ��dim(����ά��)
		KALDI_ASSERT(dim > 0 && dim < 10000);
		std::string model_filename = po.GetArg(3);
		//std::string tree_filename = po.GetArg(4);

		// ����dimά�ķ����ֵ���������ֵ
		Vector<BaseFloat> glob_inv_var(dim);
		glob_inv_var.Set(1.0);
		Vector<BaseFloat> glob_mean(dim);
		glob_mean.Set(1.0);

		if (train_feats != "") {
			double count = 0.0;
			Vector<double> var_stats(dim);
			Vector<double> mean_stats(dim);
			// typedef  SequentialTableReader<KaldiObjectHolder<Matrix<double> > >  SequentialDoubleMatrixReader;
			// SequentialDoubleMatrixReader��ģ����SequentialTableReaderȷ������֮����࣬feat_reader������һ��ʵ��
			// ���԰�Table����һ��ͨ�õ�map����list
			SequentialDoubleMatrixReader feat_reader(train_feats); // train_feats�д�Ų����ļ�������������ǰʮ���ļ���
			for (; !feat_reader.Done(); feat_reader.Next()) { // ˳�����feat_reader�ࣨ��ʵ��һ��������
				const Matrix<double> &mat = feat_reader.Value();
				// ˳���ȡ�����е�values������subfeats��ŵ�ĳ��valueΪ�����ļ�����ȡ������
				// һϵ�������������ɵ����������з�֮��ÿ��frame��Ӧһ������������
				for (int32 i = 0; i < mat.NumRows(); i++) { // ����ǰ��������ÿ��д�뵽��ֵ�ͷ����ͳ������
					count += 1.0;
					var_stats.AddVec2(1.0, mat.Row(i)); // ��ÿ��ƽ�����
					mean_stats.AddVec(1.0, mat.Row(i)); // ��ÿ�����
				}
			}
			if (count == 0) { KALDI_ERR << "no features were seen."; } // count������������������ά����
			var_stats.Scale(1.0 / count); // Scale��������ǰ����������Ԫ�س���ĳ�������������𵽹�һ�����ã�ƽ���͵ľ�ֵ��
			mean_stats.Scale(1.0 / count); // ������������ÿ����ͺ�ľ�ֵ
			var_stats.AddVec2(-1.0, mean_stats); // AddVec2����ִ��*this = *this + alpha * rv^2������=����-��ֵ^2��
			if (var_stats.Min() <= 0.0)
				KALDI_ERR << "bad variance";
			var_stats.InvertElements(); // InvertElements����������Ԫ������
			glob_inv_var.CopyFromVec(var_stats); // ���õ��Ľ�����Ƶ�ȫ�־�ֵ��������
			glob_mean.CopyFromVec(mean_stats);
		}

		HmmTopology_2D topo;
		bool binary_in;
		// Kaldi���ݿ�ͷ�Ƿ�"\0B"���ж��Ƕ����ƻ����ı���ʽ��ͬʱ׼����һ����
		Input ki(topo_filename, &binary_in); // topo_filename��š�$lang/topo��
		topo.Read(ki.Stream(), binary_in); // �������ȡ����HmmTopology��Ķ���topo

		const std::vector<int32> &phones = topo.GetPhones(); // GetPhones������topo�з�������õ������б�

		std::vector<int32> phone2num_pdf_classes(1 + phones.back()); // back���������������һ��Ԫ�ص�����
		// ����phone�д洢������1~200����phone2num_pdf_classes�����˳���Ϊ201������

		int32 num_pdfs = 0;
		int32 num_pdfs_for_each_phone = 0;
		for (size_t i = 0; i < phones.size(); i++) // ��1~end��phone2num_pdf_classes���и�ֵ
		{
			num_pdfs_for_each_phone = topo.NumPdfClasses(phones[i]);
			num_pdfs += num_pdfs_for_each_phone;
			phone2num_pdf_classes[phones[i]] = num_pdfs_for_each_phone; // NumPdfClasses�������ض�Ӧ��������ص�pdf����
		}
		/*
		// ����ÿ�� phone ���ض�Ӧ pdf ���������� ContextDependency (������)����
		// Now the tree [not really a tree at this point]:
		ContextDependency *ctx_dep = NULL;
		if (shared_phones_rxfilename == "") {  // No sharing of phones: standard approach.
			ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
		}
		else {
			std::vector<std::vector<int32> > shared_phones;
			ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones); // ��sets.int�ļ������ά����shared_phones��
			// ReadSharedPhonesList crashes on error.
			ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);
		}
		// ��ȡ���� pdfs ���� = phones * ÿ�� phone ���е� pdfclass ����
		int32 num_pdfs = ctx_dep->NumPdfs();
		*/

		// ��������ͳ�Ƴ��Ľ�������� DiagGmm ��ʼ��ģ��
		AmDiagGmm am_gmm; // AmDiagGmm������Ϊ��gmm��ɵ�����
		DiagGmm gmm;
		gmm.Resize(1, dim); // Resize arrays to this dim. Does not initialize data.
		{  // Initialize the gmm.
			Matrix<BaseFloat> inv_var(1, dim);
			inv_var.Row(0).CopyFromVec(glob_inv_var); // ��ȫ�ַ�������������ĵ�һ��
			Matrix<BaseFloat> mu(1, dim);
			mu.Row(0).CopyFromVec(glob_mean);
			Vector<BaseFloat> weights(1);
			weights.Set(1.0);
			gmm.SetInvVarsAndMeans(inv_var, mu); // ���¾�ֵ�����DiagGmm���ʵ��gmm��
			gmm.SetWeights(weights);
			gmm.ComputeGconsts(); // ����Gconsts��д��gmm
		}

		// ��ÿ�� pdf ����ʼ��Ϊ���������� gmm
		for (int i = 0; i < num_pdfs; i++)
			am_gmm.AddPdf(gmm); // ��gmms��д�뵽am_gmm�ڣ�������PDFNUM����ֵ

		// ��� perturb_factor ����
		if (perturb_factor != 0.0) { // ��perturb_factor���ⲿָ��ֵ
			for (int i = 0; i < num_pdfs; i++)
				// Perturbs���Ŷ����� the component means with a random vector����������� multiplied by the perturb factor.
				am_gmm.GetPdf(i).Perturb(perturb_factor);
		}

		// �� ContextDependency �� topo �ϲ�Ϊһ��ģ���ļ���������
		// Now the transition model:
		TransitionModel_2D trans_model_2D(topo);

		{
			Output ko(model_filename, binary); // ��model_filename�����ļ��������ko��binaryĬ��Ϊtrue������������ָ��
			trans_model_2D.Write(ko.Stream(), binary); // д��trans_model
			am_gmm.Write(ko.Stream(), binary); // д��am_gmm
		}

		/*
		// ��ContextDependency��Ϊ�������ļ�
		// Now write the tree.
		ctx_dep->Write(Output(tree_filename, binary).Stream(),
					   binary);

		delete ctx_dep;
		*/

		return 0;
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}

